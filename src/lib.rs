//! Game-neutral scene model used by the viewer, database, and navigation code.
//!
//! Importers are adapters: they translate a source's native map and live state
//! into this model. The contract has no knowledge of pre-transformation names,
//! protocols, or file formats.

mod hta;
mod pathfinding;
mod store;

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub use hta::{
    CoarseView, NavConfig, NavShared, NavStatus, Navigator, Threat, WalkView,
    DEFAULT_SENSE_RADIUS,
};
pub use pathfinding::{find_path, PathOptions};
pub use store::SceneStore;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub metadata: SceneMetadata,
    pub grid: NavGrid,
    pub entities: Vec<Entity>,
    pub prefab_collisions: Vec<PrefabCollision>,
    pub placements: Vec<PrefabPlacement>,
}

impl Scene {
    pub fn validate(&self) -> Result<()> {
        self.grid.validate()?;
        if self.metadata.schema_version != SCHEMA_VERSION {
            bail!(
                "scene schema {} is not supported (expected {})",
                self.metadata.schema_version,
                SCHEMA_VERSION
            );
        }
        for prefab in &self.prefab_collisions {
            prefab.validate()?;
        }
        let mut prefab_names = std::collections::BTreeSet::new();
        for prefab in &self.prefab_collisions {
            if !prefab_names.insert(prefab.name.as_str()) {
                bail!("duplicate prefab collision definition {:?}", prefab.name);
            }
        }
        let mut entity_ids = std::collections::BTreeSet::new();
        for entity in &self.entities {
            if !entity_ids.insert(entity.id.as_str()) {
                bail!("duplicate entity id {:?}", entity.id);
            }
            if !entity.position.into_iter().all(f64::is_finite) {
                bail!("entity {:?} has a non-finite position", entity.id);
            }
            match &entity.collider {
                Some(Collider::Circle { radius }) if !radius.is_finite() || *radius <= 0.0 => {
                    bail!("entity {:?} has an invalid circle collider", entity.id)
                }
                Some(Collider::Box { half_extents })
                    if half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0) =>
                {
                    bail!("entity {:?} has an invalid box collider", entity.id)
                }
                _ => {}
            }
            if entity
                .ranges
                .iter()
                .any(|range| !range.radius.is_finite() || range.radius < 0.0)
            {
                bail!("entity {:?} has an invalid range", entity.id);
            }
        }
        if let Some(unknown) = self
            .placements
            .iter()
            .find(|placement| !prefab_names.contains(placement.prefab.as_str()))
        {
            bail!(
                "placement references missing prefab collision definition {:?}",
                unknown.prefab
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneMetadata {
    pub schema_version: u32,
    pub scene_id: String,
    pub label: String,
    pub captured_at_unix_ms: i64,
    pub source: String,
    pub source_revision: Option<String>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NavGrid {
    pub width: usize,
    pub height: usize,
    pub origin: [f64; 2],
    pub cell_size: f64,
    /// One byte per cell, row-major: zero is traversable, non-zero is blocked.
    pub blocked: Vec<u8>,
    /// One generic shape identifier per cell. Values resolve through `shapes`.
    pub shape_ids: Vec<u8>,
    pub shapes: BTreeMap<u8, CellPrimitive>,
}

impl NavGrid {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.cell_size <= 0.0 {
            bail!("invalid grid geometry");
        }
        let cells = self.width * self.height;
        if self.blocked.len() != cells || self.shape_ids.len() != cells {
            bail!(
                "grid arrays must contain {cells} cells, got blocked={} shapes={}",
                self.blocked.len(),
                self.shape_ids.len()
            );
        }
        if let Some(missing) =
            self.shape_ids
                .iter()
                .zip(&self.blocked)
                .find_map(|(&shape, &blocked)| {
                    (blocked != 0 && !self.shapes.contains_key(&shape)).then_some(shape)
                })
        {
            bail!("blocked cell references missing shape {missing}");
        }
        Ok(())
    }

    pub fn index(&self, col: usize, row: usize) -> usize {
        row * self.width + col
    }

    pub fn in_bounds(&self, col: isize, row: isize) -> bool {
        col >= 0 && row >= 0 && col < self.width as isize && row < self.height as isize
    }

    pub fn cell(&self, world: [f64; 2]) -> Option<(usize, usize)> {
        let col = ((world[0] - self.origin[0]) / self.cell_size).floor() as isize;
        let row = ((world[1] - self.origin[1]) / self.cell_size).floor() as isize;
        self.in_bounds(col, row)
            .then_some((col as usize, row as usize))
    }

    pub fn center(&self, col: usize, row: usize) -> [f64; 2] {
        [
            self.origin[0] + (col as f64 + 0.5) * self.cell_size,
            self.origin[1] + (row as f64 + 0.5) * self.cell_size,
        ]
    }

    pub fn bounds(&self) -> [[f64; 2]; 2] {
        [
            self.origin,
            [
                self.origin[0] + self.width as f64 * self.cell_size,
                self.origin[1] + self.height as f64 * self.cell_size,
            ],
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CellPrimitive {
    Full,
    Disc { radius: f64 },
    Polygon { points: Vec<[f64; 2]> },
    Edge { from: [f64; 2], to: [f64; 2] },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    /// Free-form semantic role such as `agent`, `resource`, or `actor`.
    pub category: String,
    pub label: String,
    pub position: [f64; 2],
    pub heading_degrees: Option<f64>,
    pub collider: Option<Collider>,
    #[serde(default)]
    pub ranges: Vec<RangeRing>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Collider {
    Circle { radius: f64 },
    Box { half_extents: [f64; 2] },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RangeRing {
    pub role: String,
    pub label: String,
    pub radius: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefabCollision {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub offset: [f64; 2],
    pub cell_size: f64,
    pub blocked: Vec<u8>,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

impl PrefabCollision {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.cell_size <= 0.0
            || self.blocked.len() != self.width * self.height
        {
            bail!("invalid prefab collision definition {:?}", self.name);
        }
        if self.blocked.iter().all(|value| *value == 0) {
            bail!(
                "prefab collision definition {:?} has no collision",
                self.name
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefabPlacement {
    pub prefab: String,
    pub position: [f64; 2],
    pub yaw_degrees: f64,
    pub source: String,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}
