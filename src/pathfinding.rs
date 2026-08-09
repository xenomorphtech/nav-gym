use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{Collider, Entity, NavGrid, Scene};

#[derive(Clone, Copy, Debug)]
pub struct PathOptions {
    pub include_dynamic_colliders: bool,
    pub clearance: f64,
}

impl Default for PathOptions {
    fn default() -> Self {
        Self {
            include_dynamic_colliders: true,
            clearance: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct QueueNode {
    index: usize,
    score: f64,
}

impl Eq for QueueNode {}

impl PartialEq for QueueNode {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score == other.score
    }
}

impl Ord for QueueNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for QueueNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn find_path(
    scene: &Scene,
    start: [f64; 2],
    goal: [f64; 2],
    options: PathOptions,
) -> Option<Vec<[f64; 2]>> {
    let grid = &scene.grid;
    let start = grid.cell(start)?;
    let goal = grid.cell(goal)?;
    let blocked = effective_blocked(grid, &scene.entities, options);
    let start_index = grid.index(start.0, start.1);
    let goal_index = grid.index(goal.0, goal.1);
    if blocked[start_index] || blocked[goal_index] {
        return None;
    }

    let cells = grid.width * grid.height;
    let mut distance = vec![f64::INFINITY; cells];
    let mut previous = vec![usize::MAX; cells];
    let mut open = BinaryHeap::new();
    distance[start_index] = 0.0;
    open.push(QueueNode {
        index: start_index,
        score: heuristic(start, goal),
    });

    const NEIGHBORS: [(isize, isize, f64); 8] = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, std::f64::consts::SQRT_2),
        (-1, 1, std::f64::consts::SQRT_2),
        (1, -1, std::f64::consts::SQRT_2),
        (1, 1, std::f64::consts::SQRT_2),
    ];
    while let Some(current) = open.pop() {
        if current.index == goal_index {
            break;
        }
        let col = current.index % grid.width;
        let row = current.index / grid.width;
        for (dx, dz, cost) in NEIGHBORS {
            let next_col = col as isize + dx;
            let next_row = row as isize + dz;
            if !grid.in_bounds(next_col, next_row) {
                continue;
            }
            let next_col = next_col as usize;
            let next_row = next_row as usize;
            let next = grid.index(next_col, next_row);
            if blocked[next] {
                continue;
            }
            if dx != 0 && dz != 0 {
                let side_a = grid.index((col as isize + dx) as usize, row);
                let side_b = grid.index(col, (row as isize + dz) as usize);
                if blocked[side_a] || blocked[side_b] {
                    continue;
                }
            }
            let candidate = distance[current.index] + cost;
            if candidate >= distance[next] {
                continue;
            }
            distance[next] = candidate;
            previous[next] = current.index;
            open.push(QueueNode {
                index: next,
                score: candidate + heuristic((next_col, next_row), goal),
            });
        }
    }
    if previous[goal_index] == usize::MAX && start_index != goal_index {
        return None;
    }
    let mut indices = vec![goal_index];
    while *indices.last()? != start_index {
        let prior = previous[*indices.last()?];
        if prior == usize::MAX {
            return None;
        }
        indices.push(prior);
    }
    indices.reverse();
    Some(
        indices
            .into_iter()
            .map(|index| grid.center(index % grid.width, index / grid.width))
            .collect(),
    )
}

fn heuristic(left: (usize, usize), right: (usize, usize)) -> f64 {
    let dx = left.0.abs_diff(right.0) as f64;
    let dz = left.1.abs_diff(right.1) as f64;
    dx.max(dz) + (std::f64::consts::SQRT_2 - 1.0) * dx.min(dz)
}

fn effective_blocked(grid: &NavGrid, entities: &[Entity], options: PathOptions) -> Vec<bool> {
    let mut result = grid
        .blocked
        .iter()
        .map(|&value| value != 0)
        .collect::<Vec<_>>();
    if options.clearance > 0.0 {
        let original = result.clone();
        let radius = (options.clearance / grid.cell_size).ceil() as isize;
        for row in 0..grid.height {
            for col in 0..grid.width {
                if !original[grid.index(col, row)] {
                    continue;
                }
                for dz in -radius..=radius {
                    for dx in -radius..=radius {
                        if (dx * dx + dz * dz) as f64 > (radius * radius) as f64 {
                            continue;
                        }
                        let x = col as isize + dx;
                        let z = row as isize + dz;
                        if grid.in_bounds(x, z) {
                            result[grid.index(x as usize, z as usize)] = true;
                        }
                    }
                }
            }
        }
    }
    if options.include_dynamic_colliders {
        for entity in entities {
            if entity.category == "agent" {
                continue;
            }
            let Some(collider) = &entity.collider else {
                continue;
            };
            rasterize_entity(
                grid,
                &mut result,
                entity.position,
                collider,
                options.clearance,
            );
        }
    }
    result
}

fn rasterize_entity(
    grid: &NavGrid,
    blocked: &mut [bool],
    position: [f64; 2],
    collider: &Collider,
    clearance: f64,
) {
    let (half_x, half_z) = match collider {
        Collider::Circle { radius } => (radius + clearance, radius + clearance),
        Collider::Box { half_extents } => {
            (half_extents[0] + clearance, half_extents[1] + clearance)
        }
    };
    let clamped_cell = |world: [f64; 2]| {
        let col = ((world[0] - grid.origin[0]) / grid.cell_size).floor() as isize;
        let row = ((world[1] - grid.origin[1]) / grid.cell_size).floor() as isize;
        (
            col.clamp(0, grid.width as isize - 1) as usize,
            row.clamp(0, grid.height as isize - 1) as usize,
        )
    };
    let min = clamped_cell([
        position[0] - half_x - grid.cell_size,
        position[1] - half_z - grid.cell_size,
    ]);
    let max = clamped_cell([
        position[0] + half_x + grid.cell_size,
        position[1] + half_z + grid.cell_size,
    ]);
    for row in min.1..=max.1 {
        for col in min.0..=max.0 {
            let center = grid.center(col, row);
            let hit = match collider {
                Collider::Circle { radius } => {
                    let radius = radius + clearance + grid.cell_size * 0.5;
                    (center[0] - position[0]).powi(2) + (center[1] - position[1]).powi(2)
                        <= radius.powi(2)
                }
                Collider::Box { half_extents } => {
                    (center[0] - position[0]).abs()
                        <= half_extents[0] + clearance + grid.cell_size * 0.5
                        && (center[1] - position[1]).abs()
                            <= half_extents[1] + clearance + grid.cell_size * 0.5
                }
            };
            if hit {
                blocked[grid.index(col, row)] = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{CellPrimitive, SceneMetadata, SCHEMA_VERSION};

    #[test]
    fn astar_routes_through_gap_without_corner_cutting() {
        let mut blocked = vec![0; 25];
        for row in 0..5 {
            if row != 2 {
                blocked[row * 5 + 2] = 1;
            }
        }
        let scene = Scene {
            metadata: SceneMetadata {
                schema_version: SCHEMA_VERSION,
                scene_id: "test".into(),
                label: "test".into(),
                captured_at_unix_ms: 0,
                source: "test".into(),
                source_revision: None,
                attributes: BTreeMap::new(),
            },
            grid: NavGrid {
                width: 5,
                height: 5,
                origin: [0.0, 0.0],
                cell_size: 1.0,
                blocked,
                shape_ids: vec![0; 25],
                shapes: BTreeMap::from([(0, CellPrimitive::Full)]),
            },
            entities: vec![],
            prefab_collisions: vec![],
            placements: vec![],
        };
        let path = find_path(&scene, [0.5, 0.5], [4.5, 4.5], PathOptions::default()).unwrap();
        assert!(path.contains(&[2.5, 2.5]));
    }
}
