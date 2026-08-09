//! Integration check on the real captured scene: the HTA* navigator must
//! reach ring goals around the agent without its interpolated motion ever
//! clipping collision terrain (0.75-unit clearance disc, checked with exact
//! segment-to-rectangle geometry against the raw grid) or entering an actor's
//! awareness radius.

use nav_scene::{NavConfig, NavShared, NavStatus, Navigator, SceneStore, Threat, WalkView};

fn distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)).sqrt()
}

fn point_segment_distance(from: [f64; 2], to: [f64; 2], point: [f64; 2]) -> f64 {
    let delta = [to[0] - from[0], to[1] - from[1]];
    let length_sq = delta[0] * delta[0] + delta[1] * delta[1];
    let t = if length_sq <= f64::EPSILON {
        0.0
    } else {
        (((point[0] - from[0]) * delta[0] + (point[1] - from[1]) * delta[1]) / length_sq)
            .clamp(0.0, 1.0)
    };
    distance([from[0] + delta[0] * t, from[1] + delta[1] * t], point)
}

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn segments_cross(a1: [f64; 2], a2: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> bool {
    let d1 = orient(b1, b2, a1);
    let d2 = orient(b1, b2, a2);
    let d3 = orient(a1, a2, b1);
    let d4 = orient(a1, a2, b2);
    (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
}

fn seg_seg_distance(a1: [f64; 2], a2: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> f64 {
    if segments_cross(a1, a2, b1, b2) {
        return 0.0;
    }
    point_segment_distance(a1, a2, b1)
        .min(point_segment_distance(a1, a2, b2))
        .min(point_segment_distance(b1, b2, a1))
        .min(point_segment_distance(b1, b2, a2))
}

fn segment_rect_distance(from: [f64; 2], to: [f64; 2], min: [f64; 2], max: [f64; 2]) -> f64 {
    let inside =
        |p: [f64; 2]| p[0] >= min[0] && p[0] <= max[0] && p[1] >= min[1] && p[1] <= max[1];
    if inside(from) || inside(to) {
        return 0.0;
    }
    let corners = [
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ];
    (0..4)
        .map(|index| seg_seg_distance(from, to, corners[index], corners[(index + 1) % 4]))
        .fold(f64::INFINITY, f64::min)
}

fn assert_clearance(raw: &WalkView, from: [f64; 2], to: [f64; 2], radius: f64) {
    let cell = raw.cell_size;
    let reach = (radius / cell).ceil() as isize + 1;
    let min_col = ((((from[0].min(to[0]) - raw.origin[0]) / cell).floor() as isize) - reach).max(0);
    let max_col = ((((from[0].max(to[0]) - raw.origin[0]) / cell).floor() as isize) + reach)
        .min(raw.width as isize - 1);
    let min_row = ((((from[1].min(to[1]) - raw.origin[1]) / cell).floor() as isize) - reach).max(0);
    let max_row = ((((from[1].max(to[1]) - raw.origin[1]) / cell).floor() as isize) + reach)
        .min(raw.height as isize - 1);
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if !raw.blocked(col, row) {
                continue;
            }
            let rect_min = [
                raw.origin[0] + col as f64 * cell,
                raw.origin[1] + row as f64 * cell,
            ];
            let rect_max = [rect_min[0] + cell, rect_min[1] + cell];
            let clearance = segment_rect_distance(from, to, rect_min, rect_max);
            assert!(
                clearance + 1e-6 >= radius,
                "clearance violated: {from:?} -> {to:?} passes {clearance:.3} from blocked cell \
                 ({col},{row}), need {radius}"
            );
        }
    }
}

fn nearest_open_center(view: &WalkView, world: [f64; 2]) -> Option<[f64; 2]> {
    let (col, row) = view.cell_of(world)?;
    for radius in 0..=5_isize {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                if dx.abs().max(dz.abs()) != radius {
                    continue;
                }
                let (c, r) = (col as isize + dx, row as isize + dz);
                if !view.blocked(c, r) {
                    return Some(view.center(c as usize, r as usize));
                }
            }
        }
    }
    None
}

#[test]
fn agent_navigates_ravencairn_without_collision_or_aggro() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/ravencairn.sqlite");
    let Ok(scene) = SceneStore::load(&path) else {
        eprintln!("scene database missing; skipping");
        return;
    };
    let config = NavConfig::default();
    let clearance = config.agent_radius;
    let shared = NavShared::from_grid(&scene.grid, config);
    let raw = WalkView::from_nav_grid(&scene.grid);
    let agent = scene
        .entities
        .iter()
        .find(|entity| entity.category == "agent")
        .expect("scene has an agent")
        .position;
    // The captured spawn may hug a wall closer than the clearance radius;
    // navigation starts from the nearest cell that satisfies it.
    let start = nearest_open_center(&shared.fine, agent).expect("open cell near the agent spawn");
    let threats: Vec<Threat> = scene
        .entities
        .iter()
        .filter(|entity| entity.category == "actor")
        .filter_map(|entity| {
            let radius = entity
                .ranges
                .iter()
                .filter(|range| range.role == "awareness")
                .map(|range| range.radius)
                .fold(0.0_f64, f64::max);
            (radius > 0.0).then_some(Threat {
                center: entity.position,
                radius,
            })
        })
        .collect();
    assert!(!threats.is_empty(), "expected awareness threats in the scene");
    // Discs the agent already starts inside are exempt from the never-enter
    // assertion (the navigator only guarantees it will not enter new ones).
    let started_inside: Vec<bool> = threats
        .iter()
        .map(|threat| distance(start, threat.center) <= threat.radius)
        .collect();

    let mut tried = 0;
    let mut arrived = 0;
    for spoke in 0..8 {
        let angle = spoke as f64 / 8.0 * std::f64::consts::TAU;
        let goal = [
            start[0] + angle.cos() * 140.0,
            start[1] + angle.sin() * 140.0,
        ];
        let Some((col, row)) = shared.fine.cell_of(goal) else {
            continue;
        };
        if shared.fine.blocked(col as isize, row as isize) {
            continue;
        }
        if threats
            .iter()
            .any(|threat| distance(goal, threat.center) <= threat.radius + 1.0)
        {
            continue;
        }
        tried += 1;
        let mut navigator = Navigator::new(start);
        navigator.set_goal(goal);
        let mut last_status = NavStatus::Moving;
        for _ in 0..12_000 {
            let before = navigator.position();
            let status = navigator.tick(&shared, &shared.fine, &threats, 0.05);
            let after = navigator.position();
            assert!(
                raw.segment_is_clear(before, after),
                "interpolated move {before:?} -> {after:?} crossed collision terrain"
            );
            assert_clearance(&raw, before, after, clearance);
            for (threat, &inside) in threats.iter().zip(&started_inside) {
                if !inside {
                    assert!(
                        distance(after, threat.center) > threat.radius,
                        "entered awareness radius at {after:?} (threat {:?})",
                        threat.center
                    );
                }
            }
            last_status = status;
            match status {
                NavStatus::Arrived => {
                    arrived += 1;
                    break;
                }
                NavStatus::Blocked => break,
                _ => {}
            }
        }
        println!(
            "spoke {spoke}: {last_status:?}, {:.1} units left of {:.1}",
            distance(navigator.position(), goal),
            distance(start, goal)
        );
    }
    assert!(tried > 0, "no valid ring goals around the agent");
    assert!(arrived > 0, "none of the {tried} ring goals was reached");
    println!("reached {arrived}/{tried} ring goals");
}
