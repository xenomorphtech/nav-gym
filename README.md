# Navigation Gym

This repository is a game-neutral 2D navigation workbench. It loads a
versioned SQLite scene and knows only:

- a row-major navigation grid with generic cell primitives;
- entities with arbitrary categories, circle/box colliders, and named ranges;
- reusable prefab collision definitions and world placements.

Our sensing radius is **72 world units**.

Its boundary is intentionally one-way:

```text
scenario producer -> SQLite scene contract -> navigation gym
```

There are no source-adapter, network, or pre-transformation asset dependencies.
Any producer can create scenarios as long as it implements the documented
scene contract.

## Open the viewer

```bash
cargo run --release -- data/ravencairn.sqlite
```

For CI or headless contract validation:

```bash
cargo run -- data/ravencairn.sqlite --validate-only
```

Drag to pan, use the wheel to zoom, left-click a route start, and right-click a
route goal. Static geometry and current entity colliders can be toggled
independently; actor awareness/aggro radii are generic named range rings.

Actors wander by default. Each actor retains its loaded position as an immutable
spawn center, chooses deterministic pseudo-random directions, and stays within
the configured home radius while respecting map collision. The side panel can
pause wandering, change its radius/speed, or reset every actor to its spawn.

At startup, the gym estimates mob density from the number of loaded actors
inside the 72-unit sensing radius divided by the walkable sensed area. It then
deterministically samples the observed actor archetype mix into walkable cells
outside that radius, scaled to the remaining walkable map area. Synthetic actors
exist only in memory; the source SQLite scene is never modified.

## HTA* navigation and steering

`nav_scene` ships a hierarchical A* ("HTA*") navigator (`src/hta.rs`) built to
run thousands of agents while never entering an enemy awareness radius and
never letting interpolated motion cross collision terrain. It is the default
mode in the viewer ("HTA* steering" in the pathfinding panel): left-click
teleports the agent, right-click sets a goal and the agent steers there live,
replanning around wandering actors' aggro rings.

The planner works on derived views of the walk grid:

- `WalkView` — bit-packed fine occupancy with a conservative supercover
  raycast used for path smoothing and as the per-tick movement gate. The
  planning view is dilated for the agent's collision clearance (default
  **0.75 units**): every cell containing any point closer than the clearance
  to an obstacle is closed, so a 0.75-radius disc swept along any interpolated
  segment through open cells can never clip collision terrain.
- `CoarseView` — a downsampled, optimistic occupancy (open if any fine cell is
  open, cost-weighted by obstacle density). Global routes are planned here, so
  they never trust fine geometry outside the sensing radius, where small
  obstacles may not have been streamed in yet.

Each `Navigator` plans a global corridor on the coarse view, refines it with a
windowed fine A* over the sensed neighbourhood (enemy awareness discs are hard
obstacles, rasterized per cell), string-pulls the window path, and steers along
it. Every interpolated movement step must pass the supercover raycast and stay
outside every threat disc; the gate enforces half the threat margin less than
the planner avoids, so legal plans never chatter against it. Moving threats
are handled explicitly: inside a disc that swept over the agent the gate only
admits non-inward motion (a disc can be escaped, never crossed through), and
the agent switches to reactive flee steering — recomputed every tick, so it
tracks the moving threat with zero plan staleness — until it is back outside.
When the optimistic coarse view promises a passage the fine grid does not
deliver, the failed corridor cells are penalized and the global route
reroutes; persistent failure reports `Blocked` instead of ever entering an
enemy radius. Ordered goals are snapped to the nearest legal stopping point —
a goal on a wall, inside the clearance band, off the grid, or inside an enemy
shell means "walk as close as safely possible", so short-range orders always
move the agent instead of silently refusing.

Per-agent search state is reusable, generation-stamped scratch (no clearing,
no steady-state allocation). Capacity on a 512×512 grid with 64 threats:
**2000 navigators tick in ~15 ms (~7.3 µs per agent-tick, release)** — run it
with `cargo test --release -- --ignored --nocapture`.

Tests cover: supercover corner conservatism, dilated clearance views, maze
navigation, obstacles streamed in only inside the sensing radius, enemy-radius
avoidance and refusal (`Blocked`) when every route would enter one, a live
simulation of wandering mobs whose aggro discs sweep over the agent's route
(the agent must never be inside 0.8x of any aggro radius; it stays above
1.27x), repair of over-optimistic coarse routes, narrow-corridor traversal,
and a real-scene integration test (`tests/ravencairn.rs`). Every simulated tick in these tests
asserts — with exact segment-to-rectangle geometry against the raw grid — that
the swept 0.75-unit clearance disc never clips collision terrain and that no
awareness radius is entered. Set `NAV_DEBUG=1` for replanning diagnostics.

## Database tables

- `metadata`: capture identity and source revision.
- `nav_grid`: normalized occupancy, generic shape IDs, and shape definitions.
- `entities`: only navigation-relevant live objects.
- `prefab_collisions`: local collision definitions for prefabs used by the map
  and captured collision-bearing entities.
- `prefab_placements`: collision-bearing static placements.

The database reader, pathfinder, and renderer depend only on this normalized
contract. Complete producer/consumer invariants are documented in
[`docs/scene-contract.md`](docs/scene-contract.md).
