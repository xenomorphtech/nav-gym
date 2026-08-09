use nav_scene::{Entity, Scene};

use crate::wander::position_is_walkable;

pub const SENSING_RADIUS: f64 = 72.0;

#[derive(Clone, Debug, Default)]
pub struct PopulationReport {
    pub sensing_radius: f64,
    pub observed_actors: usize,
    pub synthetic_actors: usize,
    pub density_per_square_unit: f64,
}

/// Populates the unobserved part of a scene using the empirical actor density
/// and archetype mix inside `sensing_radius` around the agent.
///
/// Loaded actors are authoritative and never replaced. Generated actors are
/// placed only in walkable cells outside the sensed circle and live only in the
/// in-memory scene.
pub fn synthetically_populate_actors(scene: &mut Scene, sensing_radius: f64) -> PopulationReport {
    let mut report = PopulationReport {
        sensing_radius,
        ..PopulationReport::default()
    };
    if !sensing_radius.is_finite() || sensing_radius <= 0.0 {
        return report;
    }
    let Some(agent) = scene
        .entities
        .iter()
        .find(|entity| entity.category == "agent")
        .map(|entity| entity.position)
    else {
        return report;
    };
    let radius_squared = sensing_radius * sensing_radius;
    let templates = scene
        .entities
        .iter()
        .filter(|entity| {
            entity.category == "actor" && squared_distance(entity.position, agent) <= radius_squared
        })
        .cloned()
        .collect::<Vec<_>>();
    report.observed_actors = templates.len();
    if templates.is_empty() {
        return report;
    }

    let grid = &scene.grid;
    let mut sensed_walkable_cells = 0_usize;
    let mut unobserved_walkable_cells = Vec::new();
    for index in 0..grid.blocked.len() {
        if grid.blocked[index] != 0 {
            continue;
        }
        let position = grid.center(index % grid.width, index / grid.width);
        if squared_distance(position, agent) <= radius_squared {
            sensed_walkable_cells += 1;
        } else {
            unobserved_walkable_cells.push(index);
        }
    }
    if sensed_walkable_cells == 0 || unobserved_walkable_cells.is_empty() {
        return report;
    }

    let cell_area = grid.cell_size * grid.cell_size;
    let sensed_walkable_area = sensed_walkable_cells as f64 * cell_area;
    report.density_per_square_unit = report.observed_actors as f64 / sensed_walkable_area;
    let unobserved_walkable_area = unobserved_walkable_cells.len() as f64 * cell_area;
    let expected_unobserved =
        (report.density_per_square_unit * unobserved_walkable_area).round() as usize;
    let known_unobserved = scene
        .entities
        .iter()
        .filter(|entity| {
            entity.category == "actor" && squared_distance(entity.position, agent) > radius_squared
        })
        .count();
    let target = expected_unobserved
        .saturating_sub(known_unobserved)
        .min(unobserved_walkable_cells.len());

    let mut random_state = seed_from_scene(scene);
    shuffle(&mut unobserved_walkable_cells, &mut random_state);
    let mut generated = Vec::with_capacity(target);
    for cell_index in unobserved_walkable_cells {
        if generated.len() == target {
            break;
        }
        let template_index = next_index(&mut random_state, templates.len());
        let template = &templates[template_index];
        let position = grid.center(cell_index % grid.width, cell_index / grid.width);
        if !position_is_walkable(grid, position, template.collider.as_ref()) {
            continue;
        }

        let mut actor: Entity = template.clone();
        actor.id = format!("actor:synthetic:{:06}", generated.len());
        actor.position = position;
        actor.heading_degrees = Some(next_unit(&mut random_state) * 360.0);
        actor
            .attributes
            .insert("synthetic".to_owned(), "true".to_owned());
        actor.attributes.insert(
            "population_model".to_owned(),
            "observed_actor_density".to_owned(),
        );
        actor
            .attributes
            .insert("sample_actor_id".to_owned(), template.id.clone());
        generated.push(actor);
    }

    report.synthetic_actors = generated.len();
    scene.entities.extend(generated);
    scene.entities.sort_by(|left, right| {
        left.category
            .cmp(&right.category)
            .then(left.id.cmp(&right.id))
    });
    report
}

fn seed_from_scene(scene: &Scene) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in scene.metadata.scene_id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash.max(1)
}

fn shuffle(values: &mut [usize], random_state: &mut u64) {
    for index in (1..values.len()).rev() {
        let swap_with = next_index(random_state, index + 1);
        values.swap(index, swap_with);
    }
}

fn next_index(random_state: &mut u64, upper_bound: usize) -> usize {
    (next_u64(random_state) % upper_bound as u64) as usize
}

fn next_unit(random_state: &mut u64) -> f64 {
    next_u64(random_state) as f64 / u64::MAX as f64
}

fn next_u64(random_state: &mut u64) -> u64 {
    let mut value = *random_state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *random_state = value;
    value
}

fn squared_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nav_scene::{CellPrimitive, Collider, NavGrid, SceneMetadata};

    use super::*;

    fn entity(id: &str, category: &str, position: [f64; 2]) -> Entity {
        Entity {
            id: id.into(),
            category: category.into(),
            label: id.into(),
            position,
            heading_degrees: None,
            collider: Some(Collider::Circle { radius: 0.2 }),
            ranges: Vec::new(),
            attributes: BTreeMap::new(),
        }
    }

    fn scene() -> Scene {
        Scene {
            metadata: SceneMetadata {
                schema_version: 1,
                scene_id: "population-test".into(),
                label: "population test".into(),
                captured_at_unix_ms: 0,
                source: "test".into(),
                source_revision: None,
                attributes: BTreeMap::new(),
            },
            grid: NavGrid {
                width: 10,
                height: 10,
                origin: [0.0, 0.0],
                cell_size: 1.0,
                blocked: vec![0; 100],
                shape_ids: vec![0; 100],
                shapes: BTreeMap::from([(0, CellPrimitive::Full)]),
            },
            entities: vec![
                entity("agent", "agent", [5.5, 5.5]),
                entity("wolf", "actor", [5.5, 6.5]),
            ],
            prefab_collisions: Vec::new(),
            placements: Vec::new(),
        }
    }

    #[test]
    fn density_populates_only_the_unobserved_walkable_area() {
        let mut scene = scene();
        let report = synthetically_populate_actors(&mut scene, 2.0);
        assert_eq!(report.observed_actors, 1);
        assert_eq!(report.synthetic_actors, 7);
        assert!((report.density_per_square_unit - 1.0 / 13.0).abs() < f64::EPSILON);
        for actor in scene
            .entities
            .iter()
            .filter(|entity| entity.attributes.contains_key("synthetic"))
        {
            assert!(squared_distance(actor.position, [5.5, 5.5]) > 4.0);
            assert!(position_is_walkable(
                &scene.grid,
                actor.position,
                actor.collider.as_ref()
            ));
        }
    }

    #[test]
    fn population_is_deterministic_for_the_same_scene() {
        let mut left = scene();
        let mut right = scene();
        let left_report = synthetically_populate_actors(&mut left, 2.0);
        let right_report = synthetically_populate_actors(&mut right, 2.0);
        assert_eq!(left_report.synthetic_actors, right_report.synthetic_actors);
        assert_eq!(left.entities, right.entities);
    }

    #[test]
    fn no_observed_actors_means_no_synthetic_population() {
        let mut scene = scene();
        scene.entities.retain(|entity| entity.category != "actor");
        let report = synthetically_populate_actors(&mut scene, 2.0);
        assert_eq!(report.observed_actors, 0);
        assert_eq!(report.synthetic_actors, 0);
    }
}
