use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::types::Type;
use rusqlite::{params, Connection};
use serde::de::DeserializeOwned;

use crate::{
    Entity, NavGrid, PrefabCollision, PrefabPlacement, Scene, SceneMetadata, SCHEMA_VERSION,
};

pub struct SceneStore;

impl SceneStore {
    pub fn save(path: impl AsRef<Path>, scene: &Scene) -> Result<()> {
        scene.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create database directory {}", parent.display()))?;
        }
        let mut connection = Connection::open(path)
            .with_context(|| format!("open scene database {}", path.display()))?;
        connection.pragma_update(None, "journal_mode", "DELETE")?;
        let transaction = connection.transaction()?;
        transaction.execute_batch(
            "
            DROP TABLE IF EXISTS metadata;
            DROP TABLE IF EXISTS nav_grid;
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS prefab_collisions;
            DROP TABLE IF EXISTS prefab_placements;

            CREATE TABLE metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                scene_id TEXT NOT NULL,
                label TEXT NOT NULL,
                captured_at_unix_ms INTEGER NOT NULL,
                source TEXT NOT NULL,
                source_revision TEXT,
                attributes_json TEXT NOT NULL
            );
            CREATE TABLE nav_grid (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                origin_x REAL NOT NULL,
                origin_z REAL NOT NULL,
                cell_size REAL NOT NULL,
                blocked BLOB NOT NULL,
                shape_ids BLOB NOT NULL,
                shapes_json TEXT NOT NULL
            );
            CREATE TABLE entities (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                label TEXT NOT NULL,
                x REAL NOT NULL,
                z REAL NOT NULL,
                heading_degrees REAL,
                collider_json TEXT,
                ranges_json TEXT NOT NULL,
                attributes_json TEXT NOT NULL
            );
            CREATE INDEX entities_category ON entities(category);
            CREATE TABLE prefab_collisions (
                name TEXT PRIMARY KEY,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                offset_x REAL NOT NULL,
                offset_z REAL NOT NULL,
                cell_size REAL NOT NULL,
                blocked BLOB NOT NULL,
                attributes_json TEXT NOT NULL
            );
            CREATE TABLE prefab_placements (
                id INTEGER PRIMARY KEY,
                prefab TEXT NOT NULL,
                x REAL NOT NULL,
                z REAL NOT NULL,
                yaw_degrees REAL NOT NULL,
                source TEXT NOT NULL,
                attributes_json TEXT NOT NULL
            );
            CREATE INDEX prefab_placements_prefab ON prefab_placements(prefab);
            ",
        )?;
        transaction.execute(
            "INSERT INTO metadata VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                scene.metadata.schema_version,
                scene.metadata.scene_id,
                scene.metadata.label,
                scene.metadata.captured_at_unix_ms,
                scene.metadata.source,
                scene.metadata.source_revision,
                serde_json::to_string(&scene.metadata.attributes)?,
            ],
        )?;
        transaction.execute(
            "INSERT INTO nav_grid VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                scene.grid.width,
                scene.grid.height,
                scene.grid.origin[0],
                scene.grid.origin[1],
                scene.grid.cell_size,
                scene.grid.blocked,
                scene.grid.shape_ids,
                serde_json::to_string(&scene.grid.shapes)?,
            ],
        )?;
        {
            let mut statement = transaction
                .prepare("INSERT INTO entities VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)")?;
            for entity in &scene.entities {
                statement.execute(params![
                    entity.id,
                    entity.category,
                    entity.label,
                    entity.position[0],
                    entity.position[1],
                    entity.heading_degrees,
                    entity
                        .collider
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()?,
                    serde_json::to_string(&entity.ranges)?,
                    serde_json::to_string(&entity.attributes)?,
                ])?;
            }
        }
        {
            let mut statement = transaction
                .prepare("INSERT INTO prefab_collisions VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)")?;
            for prefab in &scene.prefab_collisions {
                statement.execute(params![
                    prefab.name,
                    prefab.width,
                    prefab.height,
                    prefab.offset[0],
                    prefab.offset[1],
                    prefab.cell_size,
                    prefab.blocked,
                    serde_json::to_string(&prefab.attributes)?,
                ])?;
            }
        }
        {
            let mut statement = transaction.prepare(
                "INSERT INTO prefab_placements (prefab,x,z,yaw_degrees,source,attributes_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for placement in &scene.placements {
                statement.execute(params![
                    placement.prefab,
                    placement.position[0],
                    placement.position[1],
                    placement.yaw_degrees,
                    placement.source,
                    serde_json::to_string(&placement.attributes)?,
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Scene> {
        let path = path.as_ref();
        let connection = Connection::open(path)
            .with_context(|| format!("open scene database {}", path.display()))?;
        let metadata = connection.query_row(
            "SELECT schema_version,scene_id,label,captured_at_unix_ms,source,source_revision,attributes_json FROM metadata WHERE id=1",
            [],
            |row| {
                let attributes: String = row.get(6)?;
                Ok(SceneMetadata {
                    schema_version: row.get(0)?,
                    scene_id: row.get(1)?,
                    label: row.get(2)?,
                    captured_at_unix_ms: row.get(3)?,
                    source: row.get(4)?,
                    source_revision: row.get(5)?,
                    attributes: decode_json(6, &attributes)?,
                })
            },
        )?;
        if metadata.schema_version != SCHEMA_VERSION {
            anyhow::bail!(
                "database schema {} is not supported (expected {})",
                metadata.schema_version,
                SCHEMA_VERSION
            );
        }
        let grid = connection.query_row(
            "SELECT width,height,origin_x,origin_z,cell_size,blocked,shape_ids,shapes_json FROM nav_grid WHERE id=1",
            [],
            |row| {
                let shapes: String = row.get(7)?;
                Ok(NavGrid {
                    width: row.get(0)?,
                    height: row.get(1)?,
                    origin: [row.get(2)?, row.get(3)?],
                    cell_size: row.get(4)?,
                    blocked: row.get(5)?,
                    shape_ids: row.get(6)?,
                    shapes: decode_json(7, &shapes)?,
                })
            },
        )?;
        let entities = {
            let mut statement = connection.prepare(
                "SELECT id,category,label,x,z,heading_degrees,collider_json,ranges_json,attributes_json FROM entities ORDER BY category,id",
            )?;
            let rows = statement.query_map([], |row| {
                let collider: Option<String> = row.get(6)?;
                let ranges: String = row.get(7)?;
                let attributes: String = row.get(8)?;
                Ok(Entity {
                    id: row.get(0)?,
                    category: row.get(1)?,
                    label: row.get(2)?,
                    position: [row.get(3)?, row.get(4)?],
                    heading_degrees: row.get(5)?,
                    collider: collider
                        .as_deref()
                        .map(|value| decode_json(6, value))
                        .transpose()?,
                    ranges: decode_json(7, &ranges)?,
                    attributes: decode_json(8, &attributes)?,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let prefab_collisions = {
            let mut statement = connection.prepare(
                "SELECT name,width,height,offset_x,offset_z,cell_size,blocked,attributes_json FROM prefab_collisions ORDER BY name",
            )?;
            let rows = statement.query_map([], |row| {
                let attributes: String = row.get(7)?;
                Ok(PrefabCollision {
                    name: row.get(0)?,
                    width: row.get(1)?,
                    height: row.get(2)?,
                    offset: [row.get(3)?, row.get(4)?],
                    cell_size: row.get(5)?,
                    blocked: row.get(6)?,
                    attributes: decode_json(7, &attributes)?,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let placements = {
            let mut statement = connection.prepare(
                "SELECT prefab,x,z,yaw_degrees,source,attributes_json FROM prefab_placements ORDER BY id",
            )?;
            let rows = statement.query_map([], |row| {
                let attributes: String = row.get(5)?;
                Ok(PrefabPlacement {
                    prefab: row.get(0)?,
                    position: [row.get(1)?, row.get(2)?],
                    yaw_degrees: row.get(3)?,
                    source: row.get(4)?,
                    attributes: decode_json(5, &attributes)?,
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let scene = Scene {
            metadata,
            grid,
            entities,
            prefab_collisions,
            placements,
        };
        scene.validate()?;
        Ok(scene)
    }
}

fn decode_json<T: DeserializeOwned>(column: usize, value: &str) -> rusqlite::Result<T> {
    serde_json::from_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(error))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::{CellPrimitive, SCHEMA_VERSION};

    #[test]
    fn sqlite_contract_round_trips_without_source_adapter_types() {
        let scene = Scene {
            metadata: SceneMetadata {
                schema_version: SCHEMA_VERSION,
                scene_id: "roundtrip".into(),
                label: "Round trip".into(),
                captured_at_unix_ms: 7,
                source: "fixture".into(),
                source_revision: Some("1".into()),
                attributes: BTreeMap::new(),
            },
            grid: NavGrid {
                width: 2,
                height: 2,
                origin: [-1.0, -1.0],
                cell_size: 1.0,
                blocked: vec![0, 1, 0, 0],
                shape_ids: vec![0; 4],
                shapes: BTreeMap::from([(0, CellPrimitive::Full)]),
            },
            entities: vec![],
            prefab_collisions: vec![],
            placements: vec![],
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nav-scene-roundtrip-{}-{nonce}.sqlite",
            std::process::id()
        ));
        SceneStore::save(&path, &scene).unwrap();
        let loaded = SceneStore::load(&path).unwrap();
        assert_eq!(loaded, scene);
        std::fs::remove_file(path).unwrap();
    }
}
