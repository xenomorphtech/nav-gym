# Navigation Gym

This repository is a game-neutral 2D navigation workbench. It loads a
versioned SQLite scene and knows only:

- a row-major navigation grid with generic cell primitives;
- entities with arbitrary categories, circle/box colliders, and named ranges;
- reusable prefab collision definitions and world placements.

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
