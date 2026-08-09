use nav_scene::{Collider, NavGrid, Scene};

const DEFAULT_RADIUS: f64 = 7.0;
const DEFAULT_SPEED: f64 = 1.5;
const MIN_TURN_SECONDS: f64 = 0.8;
const TURN_SECONDS_SPREAD: f64 = 2.4;

#[derive(Debug)]
struct Wanderer {
    entity_index: usize,
    spawn: [f64; 2],
    spawn_heading: Option<f64>,
    direction: [f64; 2],
    turn_in: f64,
    random_state: u64,
}

#[derive(Debug)]
pub struct WanderSimulation {
    pub enabled: bool,
    pub radius: f64,
    pub speed: f64,
    wanderers: Vec<Wanderer>,
}

impl WanderSimulation {
    pub fn from_scene(scene: &Scene) -> Self {
        let wanderers = scene
            .entities
            .iter()
            .enumerate()
            .filter(|(_, entity)| entity.category == "actor")
            .map(|(entity_index, entity)| Wanderer {
                entity_index,
                spawn: entity.position,
                spawn_heading: entity.heading_degrees,
                direction: [0.0, 0.0],
                turn_in: 0.0,
                random_state: seed_from_id(&entity.id),
            })
            .collect();
        Self {
            enabled: true,
            radius: DEFAULT_RADIUS,
            speed: DEFAULT_SPEED,
            wanderers,
        }
    }

    pub fn actor_count(&self) -> usize {
        self.wanderers.len()
    }

    pub fn update(&mut self, scene: &mut Scene, dt: f64) {
        if !self.enabled || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(0.1);
        let radius = self.radius.max(0.0);
        let distance = self.speed.max(0.0) * dt;
        if radius == 0.0 || distance == 0.0 {
            return;
        }

        let grid = &scene.grid;
        let entities = &mut scene.entities;
        for wanderer in &mut self.wanderers {
            let Some(entity) = entities.get_mut(wanderer.entity_index) else {
                continue;
            };
            wanderer.turn_in -= dt;
            if wanderer.turn_in <= 0.0 {
                choose_direction(wanderer, entity.position, radius);
            }

            let candidate = [
                entity.position[0] + wanderer.direction[0] * distance,
                entity.position[1] + wanderer.direction[1] * distance,
            ];
            let current_home_distance = squared_distance(entity.position, wanderer.spawn);
            let candidate_home_distance = squared_distance(candidate, wanderer.spawn);
            let stays_home = candidate_home_distance <= radius * radius
                || candidate_home_distance < current_home_distance;

            if stays_home && position_is_walkable(grid, candidate, entity.collider.as_ref()) {
                entity.position = candidate;
                entity.heading_degrees = Some(
                    wanderer.direction[0]
                        .atan2(wanderer.direction[1])
                        .to_degrees(),
                );
            } else {
                // Pick a different direction on the next frame. Keeping the actor in
                // place for this frame makes collision rejection deterministic.
                wanderer.turn_in = 0.0;
            }
        }
    }

    pub fn reset(&mut self, scene: &mut Scene) {
        for wanderer in &mut self.wanderers {
            let Some(entity) = scene.entities.get_mut(wanderer.entity_index) else {
                continue;
            };
            entity.position = wanderer.spawn;
            entity.heading_degrees = wanderer.spawn_heading;
            wanderer.direction = [0.0, 0.0];
            wanderer.turn_in = 0.0;
        }
    }
}

fn choose_direction(wanderer: &mut Wanderer, position: [f64; 2], radius: f64) {
    let angle = next_unit(&mut wanderer.random_state) * std::f64::consts::TAU;
    let random = [angle.sin(), angle.cos()];
    let home_delta = [
        wanderer.spawn[0] - position[0],
        wanderer.spawn[1] - position[1],
    ];
    let home_distance = home_delta[0].hypot(home_delta[1]);
    wanderer.direction = if home_distance > radius * 0.7 {
        let home = [home_delta[0] / home_distance, home_delta[1] / home_distance];
        normalize([
            home[0] * 0.75 + random[0] * 0.25,
            home[1] * 0.75 + random[1] * 0.25,
        ])
    } else {
        random
    };
    wanderer.turn_in =
        MIN_TURN_SECONDS + next_unit(&mut wanderer.random_state) * TURN_SECONDS_SPREAD;
}

pub(crate) fn position_is_walkable(
    grid: &NavGrid,
    position: [f64; 2],
    collider: Option<&Collider>,
) -> bool {
    let [half_x, half_z] = match collider {
        Some(Collider::Circle { radius }) => [*radius, *radius],
        Some(Collider::Box { half_extents }) => *half_extents,
        None => [0.0, 0.0],
    };
    let bounds = grid.bounds();
    if position[0] - half_x < bounds[0][0]
        || position[0] + half_x > bounds[1][0]
        || position[1] - half_z < bounds[0][1]
        || position[1] + half_z > bounds[1][1]
    {
        return false;
    }

    let min_col = (((position[0] - half_x - grid.origin[0]) / grid.cell_size).floor() as isize)
        .clamp(0, grid.width as isize - 1) as usize;
    let max_col = (((position[0] + half_x - grid.origin[0]) / grid.cell_size).floor() as isize)
        .clamp(0, grid.width as isize - 1) as usize;
    let min_row = (((position[1] - half_z - grid.origin[1]) / grid.cell_size).floor() as isize)
        .clamp(0, grid.height as isize - 1) as usize;
    let max_row = (((position[1] + half_z - grid.origin[1]) / grid.cell_size).floor() as isize)
        .clamp(0, grid.height as isize - 1) as usize;

    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if grid.blocked[grid.index(col, row)] == 0 {
                continue;
            }
            let cell_min = [
                grid.origin[0] + col as f64 * grid.cell_size,
                grid.origin[1] + row as f64 * grid.cell_size,
            ];
            let cell_max = [cell_min[0] + grid.cell_size, cell_min[1] + grid.cell_size];
            let intersects = match collider {
                Some(Collider::Circle { radius }) => {
                    let nearest = [
                        position[0].clamp(cell_min[0], cell_max[0]),
                        position[1].clamp(cell_min[1], cell_max[1]),
                    ];
                    squared_distance(position, nearest) <= radius * radius
                }
                Some(Collider::Box { half_extents }) => {
                    position[0] + half_extents[0] > cell_min[0]
                        && position[0] - half_extents[0] < cell_max[0]
                        && position[1] + half_extents[1] > cell_min[1]
                        && position[1] - half_extents[1] < cell_max[1]
                }
                None => true,
            };
            if intersects {
                return false;
            }
        }
    }
    true
}

fn normalize(value: [f64; 2]) -> [f64; 2] {
    let length = value[0].hypot(value[1]);
    if length > f64::EPSILON {
        [value[0] / length, value[1] / length]
    } else {
        [0.0, 1.0]
    }
}

fn squared_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

fn seed_from_id(id: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash.max(1)
}

fn next_unit(state: &mut u64) -> f64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value as f64 / u64::MAX as f64
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nav_scene::{Entity, NavGrid, Scene, SceneMetadata};

    use super::*;

    fn scene(blocked: Vec<u8>) -> Scene {
        Scene {
            metadata: SceneMetadata {
                schema_version: 1,
                scene_id: "wander-test".into(),
                label: "wander test".into(),
                captured_at_unix_ms: 0,
                source: "test".into(),
                source_revision: None,
                attributes: BTreeMap::new(),
            },
            grid: NavGrid {
                width: 20,
                height: 20,
                origin: [0.0, 0.0],
                cell_size: 1.0,
                blocked,
                shape_ids: vec![0; 400],
                shapes: BTreeMap::new(),
            },
            entities: vec![Entity {
                id: "actor:test".into(),
                category: "actor".into(),
                label: "test actor".into(),
                position: [10.5, 10.5],
                heading_degrees: Some(90.0),
                collider: Some(Collider::Circle { radius: 0.25 }),
                ranges: Vec::new(),
                attributes: BTreeMap::new(),
            }],
            prefab_collisions: Vec::new(),
            placements: Vec::new(),
        }
    }

    #[test]
    fn actor_stays_near_its_spawn_and_reset_restores_it() {
        let mut scene = scene(vec![0; 400]);
        let spawn = scene.entities[0].position;
        let mut simulation = WanderSimulation::from_scene(&scene);
        simulation.radius = 3.0;
        for _ in 0..10_000 {
            simulation.update(&mut scene, 0.02);
            assert!(squared_distance(scene.entities[0].position, spawn) <= 9.0 + 1e-9);
        }
        assert_ne!(scene.entities[0].position, spawn);
        simulation.reset(&mut scene);
        assert_eq!(scene.entities[0].position, spawn);
        assert_eq!(scene.entities[0].heading_degrees, Some(90.0));
    }

    #[test]
    fn collider_is_rejected_before_overlapping_a_blocked_cell() {
        let mut grid = scene(vec![0; 400]).grid;
        let blocked_index = grid.index(11, 10);
        grid.blocked[blocked_index] = 1;
        assert!(position_is_walkable(
            &grid,
            [10.5, 10.5],
            Some(&Collider::Circle { radius: 0.25 })
        ));
        assert!(!position_is_walkable(
            &grid,
            [10.8, 10.5],
            Some(&Collider::Circle { radius: 0.25 })
        ));
    }
}
