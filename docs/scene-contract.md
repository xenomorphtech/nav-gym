# Scene contract v1

The SQLite file is the only boundary between a scenario dumper and the
navigation gym. Source adapters may depend on native APIs, map loaders, and
asset formats; consumers must not.

## Producer contract

A producer constructs `nav_scene::Scene`, calls `Scene::validate`, and writes it
with `SceneStore::save`. Each dump is one immutable test scenario in practice:
use a distinct output path or explicit `scene_id` when retaining multiple
captures.

Required invariants:

- `metadata.schema_version` is `1`, and `scene_id` is stable for the scenario.
- `nav_grid.blocked` and `nav_grid.shape_ids` are row-major arrays of exactly
  `width * height` bytes.
- `blocked = 0` means traversable; every other value means blocked.
- Every blocked cell's shape ID resolves through the generic `shapes` map.
- Entity IDs and prefab names are unique.
- Collider/range dimensions are finite and non-negative as appropriate.
- Every prefab placement references a stored prefab-collision definition.
- Every prefab-collision definition contains at least one blocked cell.

The producer may attach source provenance as string attributes, but consumers
must treat attributes as opaque metadata.

## Consumer contract

Consumers call `SceneStore::load`, which checks the schema and invariants before
returning a scene. The navigation gym uses only the following neutral concepts:

- world-space `x/z` points;
- a normalized occupancy grid and cell primitives;
- free-form entity categories;
- circle/box colliders and named range rings;
- local prefab collision masks and placements.

Loading and rendering a transformed scenario must not open the producer's API,
map data, asset files, or source repository.

## Tables

- `metadata`: scenario identity, capture time, revision, provenance strings.
- `nav_grid`: geometry, occupancy, shape IDs, and generic shape definitions.
- `entities`: positions, optional colliders/ranges, and opaque attributes.
- `prefab_collisions`: reusable normalized collision masks.
- `prefab_placements`: world transforms for collision-bearing definitions.

The Rust structs are authoritative for v1. A future incompatible layout must
increment `SCHEMA_VERSION` and provide an explicit migration or a fresh dump.

