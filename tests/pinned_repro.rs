//! Regression test: agent pinned against a wall while other mobs' discs
//! cover the escape routes. The navigator used to freeze (the flee fan found
//! no admissible angle), trip the stall timer into a terminal `Blocked`, and
//! never recover — even after the mobs left and a trivial route existed. It
//! must now either work its way out (aggregate penetration gate + planned
//! escape fallback) or hold and resume once the discs move off.

use nav_scene::{NavConfig, NavShared, NavStatus, Navigator, Threat, WalkView};

fn distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    ((left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)).sqrt()
}

#[test]
fn pinned_against_wall_with_mobs_on_escape_routes() {
    // Solid wall across rows 0..=3; everything else open.
    let cells: Vec<u8> = (0..48 * 32)
        .map(|index| u8::from(index / 48 <= 3))
        .collect();
    let truth = WalkView::from_blocked(48, 32, [0.0, 0.0], 1.0, &cells);
    let config = NavConfig {
        coarse_factor: 4,
        sense_radius: 12.0,
        speed: 4.0,
        replan_seconds: 0.2,
        max_fail_streak: 24,
        ..NavConfig::default()
    };
    let shared = NavShared::from_view(truth.clone(), config);

    // Agent right under the wall's clearance band, ordered to the open south.
    let start = [24.5, 5.5];
    let goal = [24.5, 28.5];
    let mut navigator = Navigator::new(start);
    navigator.set_goal(goal);

    // Mob A pins from the south; mobs B and C sit on the east/west escape
    // routes. All three margin-inflated discs (4.0 + 0.5 + 0.75) contain the
    // agent, the wall closes the north.
    let pinned_threats = vec![
        Threat {
            center: [24.5, 10.0],
            radius: 4.0,
        },
        Threat {
            center: [20.5, 5.5],
            radius: 4.0,
        },
        Threat {
            center: [28.5, 5.5],
            radius: 4.0,
        },
    ];

    let dt = 0.05;
    let mut total_moved = 0.0;
    let mut status = NavStatus::Moving;
    for tick in 0..200 {
        let before = navigator.position();
        status = navigator.tick(&shared, &shared.fine, &pinned_threats, dt);
        total_moved += distance(before, navigator.position());
        if status != NavStatus::Moving {
            println!(
                "phase 1: status {status:?} after tick {tick} ({}s)",
                tick as f64 * dt
            );
            break;
        }
    }
    println!(
        "phase 1 (pinned): status {status:?}, moved {total_moved:.3} units total, position {:?}",
        navigator.position()
    );

    // Mobs wander away: no threats at all any more, route to goal is trivial.
    let mut recovered_moved = 0.0;
    for _ in 0..2000 {
        let before = navigator.position();
        status = navigator.tick(&shared, &shared.fine, &[], dt);
        recovered_moved += distance(before, navigator.position());
        if status == NavStatus::Arrived {
            break;
        }
    }
    println!(
        "phase 2 (mobs gone): status {status:?}, moved {recovered_moved:.3} units, position {:?}",
        navigator.position()
    );

    assert_eq!(
        status,
        NavStatus::Arrived,
        "agent should escape once mobs leave; got {status:?} at {:?}",
        navigator.position()
    );
}
