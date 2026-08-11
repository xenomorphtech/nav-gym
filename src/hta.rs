//! Hierarchical A* ("HTA*") navigation with steering for large actor counts.
//!
//! The planner works on derived views of the scene walk grid:
//!
//! - [`WalkView`]: bit-packed fine occupancy (optionally dilated for
//!   clearance) with a conservative supercover raycast used for smoothing and
//!   for the per-tick movement gate.
//! - [`CoarseView`]: a downsampled, optimistic occupancy — a coarse cell is
//!   open if any fine cell inside it is open — with a density cost so global
//!   routes prefer open terrain without hard-committing to fine geometry that
//!   may not be streamed in yet.
//!
//! A [`Navigator`] plans a global corridor on the coarse view, refines it with
//! a windowed fine A* around the agent (the sensed neighbourhood), string-pulls
//! the window path, and steers along it. Every movement step is validated with
//! the supercover raycast so interpolated motion never crosses blocked cells,
//! and threat discs (enemy awareness radii) are hard obstacles for both the
//! refinement and the movement gate. When the optimistic coarse view promised a
//! passage the fine grid does not deliver, the failed corridor cells are
//! penalized and the global route is rebuilt around them.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::NavGrid;

/// Default sensing radius matching the gym's population model.
pub const DEFAULT_SENSE_RADIUS: f64 = 72.0;

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

#[derive(Clone, Copy, Debug)]
pub struct NavConfig {
    /// Fine cells per coarse cell edge in the derived global view.
    pub coarse_factor: usize,
    /// Radius around the agent whose fine geometry is trusted and re-planned.
    pub sense_radius: f64,
    /// Agent body radius; dilates the fine planning view when positive.
    pub agent_radius: f64,
    /// Extra distance kept outside every threat radius.
    pub threat_margin: f64,
    /// Steering speed in world units per second.
    pub speed: f64,
    /// Cost multiplier applied to coarse cells by blocked-cell fraction.
    pub density_cost: f64,
    /// Seconds between periodic local replans while moving.
    pub replan_seconds: f64,
    /// Consecutive failed refinements before reporting `Blocked`.
    pub max_fail_streak: u32,
    /// Settled-node cap for one windowed fine search.
    pub local_expansion_cap: usize,
    /// Cost charged to a coarse cell whose promised passage failed to refine.
    pub fail_penalty: f64,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            coarse_factor: 8,
            sense_radius: DEFAULT_SENSE_RADIUS,
            agent_radius: 0.75,
            threat_margin: 0.5,
            speed: 3.0,
            density_cost: 6.0,
            replan_seconds: 0.5,
            max_fail_streak: 12,
            local_expansion_cap: 6000,
            fail_penalty: 300.0,
        }
    }
}

/// Bit-packed occupancy view of a walk grid. Out-of-bounds reads are blocked.
#[derive(Clone, Debug)]
pub struct WalkView {
    pub width: usize,
    pub height: usize,
    pub origin: [f64; 2],
    pub cell_size: f64,
    words: Vec<u64>,
}

impl WalkView {
    pub fn from_nav_grid(grid: &NavGrid) -> Self {
        Self::from_blocked(grid.width, grid.height, grid.origin, grid.cell_size, &grid.blocked)
    }

    pub fn from_blocked(
        width: usize,
        height: usize,
        origin: [f64; 2],
        cell_size: f64,
        blocked: &[u8],
    ) -> Self {
        let mut words = vec![0_u64; (width * height).div_ceil(64)];
        for (index, &value) in blocked.iter().enumerate() {
            if value != 0 {
                words[index / 64] |= 1 << (index % 64);
            }
        }
        Self {
            width,
            height,
            origin,
            cell_size,
            words,
        }
    }

    pub fn blocked(&self, col: isize, row: isize) -> bool {
        if col < 0 || row < 0 || col >= self.width as isize || row >= self.height as isize {
            return true;
        }
        let index = row as usize * self.width + col as usize;
        self.words[index / 64] & (1 << (index % 64)) != 0
    }

    pub fn set_blocked(&mut self, col: usize, row: usize, blocked: bool) {
        let index = row * self.width + col;
        if blocked {
            self.words[index / 64] |= 1 << (index % 64);
        } else {
            self.words[index / 64] &= !(1 << (index % 64));
        }
    }

    pub fn cell_of(&self, world: [f64; 2]) -> Option<(usize, usize)> {
        let col = ((world[0] - self.origin[0]) / self.cell_size).floor() as isize;
        let row = ((world[1] - self.origin[1]) / self.cell_size).floor() as isize;
        (col >= 0 && row >= 0 && col < self.width as isize && row < self.height as isize)
            .then_some((col as usize, row as usize))
    }

    pub fn center(&self, col: usize, row: usize) -> [f64; 2] {
        [
            self.origin[0] + (col as f64 + 0.5) * self.cell_size,
            self.origin[1] + (row as f64 + 0.5) * self.cell_size,
        ]
    }

    /// Derived clearance view: additionally blocks every cell that contains
    /// ANY point closer than `radius` to a blocked cell, so every position
    /// inside a cell that stays open keeps the full clearance. This is what
    /// lets an agent disc of `radius` follow interpolated segments through
    /// open cells without ever clipping collision terrain.
    pub fn dilated(&self, radius: f64) -> WalkView {
        let mut offsets = Vec::new();
        let reach = (radius / self.cell_size).ceil() as isize + 1;
        for dz in -reach..=reach {
            for dx in -reach..=reach {
                let gap = [
                    (dx.abs() - 1).max(0) as f64 * self.cell_size,
                    (dz.abs() - 1).max(0) as f64 * self.cell_size,
                ];
                if gap[0].hypot(gap[1]) < radius {
                    offsets.push((dx, dz));
                }
            }
        }
        let mut result = self.clone();
        for row in 0..self.height as isize {
            for col in 0..self.width as isize {
                if !self.blocked(col, row) {
                    continue;
                }
                for &(dx, dz) in &offsets {
                    let (ncol, nrow) = (col + dx, row + dz);
                    if ncol >= 0
                        && nrow >= 0
                        && ncol < self.width as isize
                        && nrow < self.height as isize
                    {
                        result.set_blocked(ncol as usize, nrow as usize, true);
                    }
                }
            }
        }
        result
    }

    /// Conservative supercover raycast: true when the segment stays inside the
    /// grid and never touches a blocked cell. Exact corner crossings require
    /// both adjacent cells to be open, matching the planner's no-corner-cutting
    /// rule, so a path validated cell-by-cell is also validated here.
    pub fn segment_is_clear(&self, from: [f64; 2], to: [f64; 2]) -> bool {
        self.segment_clear_from(from, to, false)
    }

    /// Like [`Self::segment_is_clear`], but `ignore_start_cell` exempts the
    /// cell containing `from` — used to let an agent escape a cell that a
    /// clearance dilation closed underneath it (e.g. a spawn against a wall).
    pub fn segment_clear_from(
        &self,
        from: [f64; 2],
        to: [f64; 2],
        ignore_start_cell: bool,
    ) -> bool {
        let sx = (from[0] - self.origin[0]) / self.cell_size;
        let sy = (from[1] - self.origin[1]) / self.cell_size;
        let ex = (to[0] - self.origin[0]) / self.cell_size;
        let ey = (to[1] - self.origin[1]) / self.cell_size;
        let mut col = sx.floor() as isize;
        let mut row = sy.floor() as isize;
        let end_col = ex.floor() as isize;
        let end_row = ey.floor() as isize;
        if !ignore_start_cell && self.blocked(col, row) {
            return false;
        }
        if ignore_start_cell && (col < 0 || row < 0) {
            return false;
        }
        let dx = ex - sx;
        let dy = ey - sy;
        let step_col: isize = if dx > 0.0 { 1 } else { -1 };
        let step_row: isize = if dy > 0.0 { 1 } else { -1 };
        let mut t_max_col = if dx != 0.0 {
            let boundary = if dx > 0.0 { col as f64 + 1.0 } else { col as f64 };
            (boundary - sx) / dx
        } else {
            f64::INFINITY
        };
        let mut t_max_row = if dy != 0.0 {
            let boundary = if dy > 0.0 { row as f64 + 1.0 } else { row as f64 };
            (boundary - sy) / dy
        } else {
            f64::INFINITY
        };
        let t_delta_col = if dx != 0.0 { (1.0 / dx).abs() } else { f64::INFINITY };
        let t_delta_row = if dy != 0.0 { (1.0 / dy).abs() } else { f64::INFINITY };
        let mut guard = (end_col - col).abs() + (end_row - row).abs() + 4;
        while (col != end_col || row != end_row) && guard > 0 {
            guard -= 1;
            if (t_max_col - t_max_row).abs() < 1e-12 {
                if self.blocked(col + step_col, row) || self.blocked(col, row + step_row) {
                    return false;
                }
                col += step_col;
                row += step_row;
                t_max_col += t_delta_col;
                t_max_row += t_delta_row;
            } else if t_max_col < t_max_row {
                col += step_col;
                t_max_col += t_delta_col;
            } else {
                row += step_row;
                t_max_row += t_delta_row;
            }
            if self.blocked(col, row) {
                return false;
            }
        }
        guard > 0
    }
}

/// Optimistic downsampled view: a coarse cell is open when any fine cell in it
/// is open, with the blocked fraction kept as a traversal-cost density.
#[derive(Clone, Debug)]
pub struct CoarseView {
    pub factor: usize,
    pub width: usize,
    pub height: usize,
    pub origin: [f64; 2],
    pub cell_size: f64,
    open: Vec<bool>,
    density: Vec<f32>,
}

impl CoarseView {
    pub fn from_fine(fine: &WalkView, factor: usize) -> Self {
        let factor = factor.max(1);
        let width = fine.width.div_ceil(factor);
        let height = fine.height.div_ceil(factor);
        let mut open = vec![false; width * height];
        let mut density = vec![0.0_f32; width * height];
        for crow in 0..height {
            for ccol in 0..width {
                let mut blocked = 0_usize;
                let mut total = 0_usize;
                for row in crow * factor..((crow + 1) * factor).min(fine.height) {
                    for col in ccol * factor..((ccol + 1) * factor).min(fine.width) {
                        total += 1;
                        if fine.blocked(col as isize, row as isize) {
                            blocked += 1;
                        }
                    }
                }
                let index = crow * width + ccol;
                open[index] = blocked < total;
                density[index] = blocked as f32 / total.max(1) as f32;
            }
        }
        Self {
            factor,
            width,
            height,
            origin: fine.origin,
            cell_size: fine.cell_size * factor as f64,
            open,
            density,
        }
    }

    pub fn is_open(&self, col: isize, row: isize) -> bool {
        col >= 0
            && row >= 0
            && col < self.width as isize
            && row < self.height as isize
            && self.open[row as usize * self.width + col as usize]
    }

    pub fn cell_of(&self, world: [f64; 2]) -> Option<(usize, usize)> {
        let col = ((world[0] - self.origin[0]) / self.cell_size).floor() as isize;
        let row = ((world[1] - self.origin[1]) / self.cell_size).floor() as isize;
        (col >= 0 && row >= 0 && col < self.width as isize && row < self.height as isize)
            .then_some((col as usize, row as usize))
    }

    pub fn center(&self, col: usize, row: usize) -> [f64; 2] {
        [
            self.origin[0] + (col as f64 + 0.5) * self.cell_size,
            self.origin[1] + (row as f64 + 0.5) * self.cell_size,
        ]
    }
}

/// Shared, immutable planning views. Build once per scene and share across all
/// navigators (it is cheap to clone the reference, never the data).
#[derive(Clone, Debug)]
pub struct NavShared {
    pub config: NavConfig,
    pub fine: WalkView,
    pub coarse: CoarseView,
}

impl NavShared {
    pub fn from_grid(grid: &NavGrid, config: NavConfig) -> Self {
        Self::from_view(WalkView::from_nav_grid(grid), config)
    }

    pub fn from_view(base: WalkView, config: NavConfig) -> Self {
        let fine = if config.agent_radius > 0.0 {
            base.dilated(config.agent_radius)
        } else {
            base
        };
        let coarse = CoarseView::from_fine(&fine, config.coarse_factor);
        Self { config, fine, coarse }
    }
}

/// An enemy awareness disc the agent must never enter.
#[derive(Clone, Copy, Debug)]
pub struct Threat {
    pub center: [f64; 2],
    pub radius: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavStatus {
    Idle,
    Moving,
    Arrived,
    Blocked,
}

#[derive(Clone, Copy)]
struct HeapNode {
    score: f32,
    index: u32,
}

impl Eq for HeapNode {}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.score == other.score
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable per-navigator search state. Arrays are generation-stamped so a new
/// search never has to clear them, and nothing is allocated after warm-up.
#[derive(Default)]
struct Scratch {
    stamp: Vec<u32>,
    closed: Vec<u32>,
    g: Vec<f32>,
    came: Vec<u32>,
    mark_stamp: Vec<u32>,
    pen_stamp: Vec<u32>,
    pen: Vec<f32>,
    generation: u32,
    mark_generation: u32,
    pen_generation: u32,
    heap: BinaryHeap<HeapNode>,
    nearby: Vec<Threat>,
    path_cells: Vec<u32>,
    points: Vec<[f64; 2]>,
}

impl Scratch {
    fn ensure(&mut self, cells: usize) {
        if self.stamp.len() < cells {
            self.stamp.resize(cells, 0);
            self.closed.resize(cells, 0);
            self.g.resize(cells, 0.0);
            self.came.resize(cells, u32::MAX);
            self.mark_stamp.resize(cells, 0);
            self.pen_stamp.resize(cells, 0);
            self.pen.resize(cells, 0.0);
        }
    }
}

struct Window {
    min_col: usize,
    min_row: usize,
    width: usize,
    height: usize,
}

impl Window {
    fn contains(&self, col: isize, row: isize) -> bool {
        col >= self.min_col as isize
            && row >= self.min_row as isize
            && col < (self.min_col + self.width) as isize
            && row < (self.min_row + self.height) as isize
    }

    fn index(&self, col: isize, row: isize) -> usize {
        (row as usize - self.min_row) * self.width + (col as usize - self.min_col)
    }
}

enum LocalOutcome {
    Planned,
    Failed { span_start: usize, span_end: usize },
}

pub struct Navigator {
    position: [f64; 2],
    goal: Option<[f64; 2]>,
    /// The ordered goal resolved to the nearest legal stopping point.
    snapped: [f64; 2],
    status: NavStatus,
    corridor: Vec<[f64; 2]>,
    corridor_next: usize,
    local: Vec<[f64; 2]>,
    local_next: usize,
    replan_in: f64,
    fail_streak: u32,
    stall_time: f64,
    retry_in: f64,
    penalties: HashMap<u32, f32>,
    scratch: Scratch,
}

impl Navigator {
    pub fn new(position: [f64; 2]) -> Self {
        Self {
            position,
            goal: None,
            snapped: position,
            status: NavStatus::Idle,
            corridor: Vec::new(),
            corridor_next: 0,
            local: Vec::new(),
            local_next: 0,
            replan_in: 0.0,
            fail_streak: 0,
            stall_time: 0.0,
            retry_in: 0.0,
            penalties: HashMap::new(),
            scratch: Scratch::default(),
        }
    }

    pub fn position(&self) -> [f64; 2] {
        self.position
    }

    pub fn goal(&self) -> Option<[f64; 2]> {
        self.goal
    }

    pub fn status(&self) -> NavStatus {
        self.status
    }

    pub fn corridor(&self) -> &[[f64; 2]] {
        &self.corridor
    }

    pub fn local_remaining(&self) -> &[[f64; 2]] {
        &self.local[self.local_next.min(self.local.len())..]
    }

    pub fn set_goal(&mut self, goal: [f64; 2]) {
        self.goal = Some(goal);
        self.snapped = goal;
        self.status = NavStatus::Moving;
        self.reset_plans();
        self.penalties.clear();
    }

    pub fn set_position(&mut self, position: [f64; 2]) {
        self.position = position;
        self.reset_plans();
        self.penalties.clear();
        if self.goal.is_some() {
            self.status = NavStatus::Moving;
        }
    }

    pub fn stop(&mut self) {
        self.goal = None;
        self.status = NavStatus::Idle;
        self.reset_plans();
    }

    /// Report `Blocked` and arm the re-probe cooldown; while the goal stands,
    /// `tick` flips back to `Moving` once the cooldown expires.
    fn block(&mut self, cfg: &NavConfig) {
        self.status = NavStatus::Blocked;
        self.retry_in = cfg.replan_seconds.max(0.05) * 10.0;
    }

    fn reset_plans(&mut self) {
        self.corridor.clear();
        self.corridor_next = 0;
        self.local.clear();
        self.local_next = 0;
        self.replan_in = 0.0;
        self.fail_streak = 0;
        self.stall_time = 0.0;
    }

    /// Advance one simulation step. `sensed` is the caller's current best
    /// knowledge of the fine grid (streamed obstacles included as they are
    /// discovered); it must share the geometry of `shared.fine`. Only the
    /// neighbourhood within `sense_radius` of the agent needs to be accurate.
    pub fn tick(
        &mut self,
        shared: &NavShared,
        sensed: &WalkView,
        threats: &[Threat],
        dt: f64,
    ) -> NavStatus {
        if !dt.is_finite() || dt <= 0.0 {
            return self.status;
        }
        if self.status != NavStatus::Moving {
            // Blocked is a report, not a terminal state: threats wander and
            // geometry streams in, so a blocked navigator with a standing
            // goal re-probes after a cooldown instead of freezing forever.
            if self.status == NavStatus::Blocked && self.goal.is_some() {
                self.retry_in -= dt;
                if self.retry_in <= 0.0 {
                    self.reset_plans();
                    self.penalties.clear();
                    self.status = NavStatus::Moving;
                }
            }
            return self.status;
        }
        let Some(goal) = self.goal else {
            self.status = NavStatus::Idle;
            return self.status;
        };
        let cfg = &shared.config;
        let inflate = cfg.threat_margin + cfg.agent_radius;

        self.scratch.nearby.clear();
        for threat in threats {
            let radius = threat.radius + inflate;
            if distance(threat.center, self.position) <= cfg.sense_radius + radius {
                self.scratch.nearby.push(Threat {
                    center: threat.center,
                    radius,
                });
            }
        }
        let inside_threat = self
            .scratch
            .nearby
            .iter()
            .any(|threat| distance_sq(self.position, threat.center) <= threat.radius * threat.radius);

        // Snapped goals are always exactly reachable open-cell points, so the
        // arrival tolerance can be tight — important for short-range orders.
        let arrive = (shared.fine.cell_size * 0.05).max(1e-3);
        if distance(self.position, self.snapped) <= arrive {
            self.status = NavStatus::Arrived;
            return self.status;
        }

        let moved = if inside_threat {
            // A disc swept over the agent. A goal-directed plan is stale
            // against a moving threat within a tick, so switch to reactive
            // flee steering recomputed every tick until the agent is out;
            // the movement gate still forbids increasing total penetration.
            self.local.clear();
            self.local_next = 0;
            self.replan_in = 0.0;
            let fled = self.flee(shared, sensed, dt);
            if fled > 1e-9 {
                fled
            } else {
                // The reactive fan has no admissible step (pinned against
                // terrain with other discs covering the fan): fall back to a
                // planned escape. Containing discs are soft in the local
                // planner, so a least-cost way out exists whenever the gate
                // admits one; if even that fails, the agent holds and the
                // Blocked retry above keeps re-probing as the discs move.
                self.replan(shared, sensed, threats, goal);
                if self.status != NavStatus::Moving {
                    return self.status;
                }
                self.steer(shared, sensed, dt)
            }
        } else {
            self.replan_in -= dt;
            if self.local_next >= self.local.len() || self.replan_in <= 0.0 {
                self.replan(shared, sensed, threats, goal);
                if self.status != NavStatus::Moving {
                    return self.status;
                }
            }
            self.steer(shared, sensed, dt)
        };
        if moved <= 1e-9 {
            self.stall_time += dt;
            if self.stall_time > cfg.replan_seconds.max(0.05) * 20.0 {
                if std::env::var_os("NAV_DEBUG").is_some() {
                    eprintln!(
                        "stall block at {:?}: local {} points (next {}), corridor {}",
                        self.position,
                        self.local.len(),
                        self.local_next,
                        self.corridor.len()
                    );
                }
                self.block(cfg);
            }
        } else {
            self.stall_time = 0.0;
        }
        if distance(self.position, self.snapped) <= arrive {
            self.status = NavStatus::Arrived;
        }
        self.status
    }

    /// Resolve the ordered goal to the nearest legal stopping point: an open
    /// cell of the sensed view outside every (margin-inflated) threat disc.
    /// A goal on a wall, inside clearance dilation, off the grid, or inside an
    /// enemy shell thus means "walk as close as safely possible" instead of
    /// refusing to move.
    fn snap_goal(
        &self,
        shared: &NavShared,
        sensed: &WalkView,
        threats: &[Threat],
        goal: [f64; 2],
    ) -> Option<[f64; 2]> {
        let inflate = shared.config.threat_margin + shared.config.agent_radius;
        let clear = |point: [f64; 2]| -> bool {
            sensed
                .cell_of(point)
                .is_some_and(|(col, row)| !sensed.blocked(col as isize, row as isize))
                && threats.iter().all(|threat| {
                    distance_sq(point, threat.center) > (threat.radius + inflate).powi(2)
                })
        };
        if clear(goal) {
            return Some(goal);
        }
        let cell = sensed.cell_size;
        let goal_col = (((goal[0] - sensed.origin[0]) / cell).floor() as isize)
            .clamp(0, sensed.width as isize - 1);
        let goal_row = (((goal[1] - sensed.origin[1]) / cell).floor() as isize)
            .clamp(0, sensed.height as isize - 1);
        for radius in 0..=16_isize {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    if dx.abs().max(dz.abs()) != radius {
                        continue;
                    }
                    let (col, row) = (goal_col + dx, goal_row + dz);
                    if col < 0
                        || row < 0
                        || col >= sensed.width as isize
                        || row >= sensed.height as isize
                    {
                        continue;
                    }
                    let center = sensed.center(col as usize, row as usize);
                    if clear(center) {
                        return Some(center);
                    }
                }
            }
        }
        None
    }

    fn replan(&mut self, shared: &NavShared, sensed: &WalkView, threats: &[Threat], goal: [f64; 2]) {
        let cfg = &shared.config;
        let Some(snapped) = self.snap_goal(shared, sensed, threats, goal) else {
            if std::env::var_os("NAV_DEBUG").is_some() {
                eprintln!("no legal stopping point near goal {goal:?}");
            }
            self.block(cfg);
            return;
        };
        self.snapped = snapped;
        if self.corridor.is_empty() && !self.plan_global(shared, threats, snapped) {
            if std::env::var_os("NAV_DEBUG").is_some() {
                eprintln!("global plan failed at {:?}", self.position);
            }
            self.block(cfg);
            return;
        }
        match self.plan_local(shared, sensed, snapped) {
            LocalOutcome::Planned => {
                self.fail_streak = 0;
                self.replan_in = cfg.replan_seconds;
            }
            LocalOutcome::Failed { span_start, span_end } => {
                self.fail_streak += 1;
                if std::env::var_os("NAV_DEBUG").is_some() {
                    eprintln!(
                        "local fail #{} at {:?}: span {span_start}..={span_end} of {} corridor points",
                        self.fail_streak,
                        self.position,
                        self.corridor.len()
                    );
                }
                // The optimistic coarse view promised a passage the fine grid
                // does not deliver: charge the failed corridor span so the next
                // global plan routes around it.
                if !self.corridor.is_empty() {
                    let end = span_end.min(self.corridor.len() - 1);
                    for index in span_start..=end {
                        if let Some((col, row)) = shared.coarse.cell_of(self.corridor[index]) {
                            let key = (row * shared.coarse.width + col) as u32;
                            *self.penalties.entry(key).or_insert(0.0) += cfg.fail_penalty as f32;
                        }
                    }
                }
                self.corridor.clear();
                self.corridor_next = 0;
                self.local.clear();
                self.local_next = 0;
                self.replan_in = cfg.replan_seconds * 0.25;
                if self.fail_streak >= cfg.max_fail_streak {
                    self.block(cfg);
                }
            }
        }
    }

    fn plan_global(&mut self, shared: &NavShared, threats: &[Threat], goal: [f64; 2]) -> bool {
        let cfg = &shared.config;
        let coarse = &shared.coarse;
        let cells = coarse.width * coarse.height;
        self.scratch.ensure(cells);
        let Some(start) = coarse
            .cell_of(self.position)
            .and_then(|cell| nearest_open_coarse(coarse, cell))
        else {
            return false;
        };
        let Some(goal_cell) = coarse
            .cell_of(goal)
            .and_then(|cell| nearest_open_coarse(coarse, cell))
        else {
            return false;
        };

        let scratch = &mut self.scratch;
        let penalties = &self.penalties;

        // Threats are soft at this level; the fine window is the hard
        // guarantee. Hard-blocking here could disconnect routes that legally
        // squeeze past a threat.
        scratch.pen_generation += 1;
        let pen_generation = scratch.pen_generation;
        let threat_pen = (10.0 * coarse.cell_size) as f32;
        let inflate = cfg.threat_margin + cfg.agent_radius + coarse.cell_size * 0.5;
        for threat in threats {
            let avoid = threat.radius + inflate;
            let min_col = (((threat.center[0] - avoid - coarse.origin[0]) / coarse.cell_size)
                .floor() as isize)
                .max(0);
            let max_col = (((threat.center[0] + avoid - coarse.origin[0]) / coarse.cell_size)
                .floor() as isize)
                .min(coarse.width as isize - 1);
            let min_row = (((threat.center[1] - avoid - coarse.origin[1]) / coarse.cell_size)
                .floor() as isize)
                .max(0);
            let max_row = (((threat.center[1] + avoid - coarse.origin[1]) / coarse.cell_size)
                .floor() as isize)
                .min(coarse.height as isize - 1);
            for row in min_row..=max_row {
                for col in min_col..=max_col {
                    let center = coarse.center(col as usize, row as usize);
                    if distance_sq(center, threat.center) <= avoid * avoid {
                        let index = row as usize * coarse.width + col as usize;
                        if scratch.pen_stamp[index] != pen_generation {
                            scratch.pen_stamp[index] = pen_generation;
                            scratch.pen[index] = 0.0;
                        }
                        scratch.pen[index] += threat_pen;
                    }
                }
            }
        }

        scratch.generation += 1;
        let generation = scratch.generation;
        scratch.heap.clear();
        let start_index = start.1 * coarse.width + start.0;
        let goal_index = goal_cell.1 * coarse.width + goal_cell.0;
        scratch.stamp[start_index] = generation;
        scratch.g[start_index] = 0.0;
        scratch.came[start_index] = u32::MAX;
        scratch.heap.push(HeapNode {
            score: 0.0,
            index: start_index as u32,
        });
        let mut found = start_index == goal_index;
        while let Some(node) = scratch.heap.pop() {
            let index = node.index as usize;
            if scratch.closed[index] == generation {
                continue;
            }
            scratch.closed[index] = generation;
            if index == goal_index {
                found = true;
                break;
            }
            let col = (index % coarse.width) as isize;
            let row = (index / coarse.width) as isize;
            for (dx, dz, base) in NEIGHBORS {
                let ncol = col + dx;
                let nrow = row + dz;
                if !coarse.is_open(ncol, nrow) {
                    continue;
                }
                if dx != 0 && dz != 0 && (!coarse.is_open(ncol, row) || !coarse.is_open(col, nrow))
                {
                    continue;
                }
                let nindex = nrow as usize * coarse.width + ncol as usize;
                let mut step = (base * coarse.cell_size) as f32
                    * (1.0 + cfg.density_cost as f32 * coarse.density[nindex]);
                if scratch.pen_stamp[nindex] == pen_generation {
                    step += scratch.pen[nindex];
                }
                if let Some(extra) = penalties.get(&(nindex as u32)) {
                    step += *extra;
                }
                if scratch.stamp[nindex] != generation {
                    scratch.stamp[nindex] = generation;
                    scratch.g[nindex] = f32::INFINITY;
                    scratch.came[nindex] = u32::MAX;
                }
                let candidate = scratch.g[index] + step;
                if candidate < scratch.g[nindex] {
                    scratch.g[nindex] = candidate;
                    scratch.came[nindex] = index as u32;
                    let h = (octile(
                        (ncol - goal_cell.0 as isize) as f64,
                        (nrow - goal_cell.1 as isize) as f64,
                    ) * coarse.cell_size) as f32;
                    scratch.heap.push(HeapNode {
                        score: candidate + h,
                        index: nindex as u32,
                    });
                }
            }
        }
        if !found {
            return false;
        }
        scratch.path_cells.clear();
        let mut cursor = goal_index;
        loop {
            scratch.path_cells.push(cursor as u32);
            if cursor == start_index {
                break;
            }
            let prior = scratch.came[cursor];
            if prior == u32::MAX {
                return false;
            }
            cursor = prior as usize;
        }
        self.corridor.clear();
        for &cell in self.scratch.path_cells.iter().rev() {
            let cell = cell as usize;
            self.corridor
                .push(coarse.center(cell % coarse.width, cell / coarse.width));
        }
        self.corridor_next = 0;
        true
    }

    fn plan_local(&mut self, shared: &NavShared, sensed: &WalkView, goal: [f64; 2]) -> LocalOutcome {
        let cfg = &shared.config;
        let cell = sensed.cell_size;
        let position = self.position;
        let failed = LocalOutcome::Failed {
            span_start: self.corridor_next,
            span_end: self.corridor_next,
        };
        let Some((acol, arow)) = sensed.cell_of(position) else {
            return failed;
        };

        // The window is the sensed neighbourhood we trust point-by-point.
        let half = ((cfg.sense_radius / cell).floor() as isize).max(2);
        let min_col = (acol as isize - half).max(0) as usize;
        let min_row = (arow as isize - half).max(0) as usize;
        let max_col = ((acol as isize + half).min(sensed.width as isize - 1)) as usize;
        let max_row = ((arow as isize + half).min(sensed.height as isize - 1)) as usize;
        let window = Window {
            min_col,
            min_row,
            width: max_col - min_col + 1,
            height: max_row - min_row + 1,
        };
        let cells = window.width * window.height;
        self.scratch.ensure(cells);

        // Monotonic corridor progress: bounded scan so the nearest point on a
        // far-side switchback can never yank progress across a wall.
        if !self.corridor.is_empty() {
            let limit = (self.corridor_next + 48).min(self.corridor.len());
            let mut best = self.corridor_next;
            let mut best_d = f64::INFINITY;
            for index in self.corridor_next..limit {
                let d = distance_sq(self.corridor[index], position);
                if d < best_d {
                    best_d = d;
                    best = index;
                }
            }
            self.corridor_next = best;
        }
        let (target_world, span_end) = if self.corridor.is_empty() {
            (goal, self.corridor_next)
        } else {
            let mut index = self.corridor_next;
            while index + 1 < self.corridor.len()
                && distance(self.corridor[index + 1], position) <= cfg.sense_radius * 0.95
            {
                index += 1;
            }
            if index + 1 == self.corridor.len() && distance(goal, position) <= cfg.sense_radius {
                (goal, index)
            } else {
                (self.corridor[index], index)
            }
        };
        let failed = LocalOutcome::Failed {
            span_start: self.corridor_next,
            span_end,
        };

        self.scratch.generation += 1;
        self.scratch.mark_generation += 1;
        self.scratch.pen_generation += 1;
        let generation = self.scratch.generation;
        let mark_generation = self.scratch.mark_generation;
        let soft_generation = self.scratch.pen_generation;
        let Scratch {
            nearby,
            mark_stamp,
            pen_stamp,
            stamp,
            closed,
            g,
            came,
            heap,
            path_cells,
            points,
            ..
        } = &mut self.scratch;
        // Rasterize threat discs into the window. Cells whose center is inside
        // the (margin-inflated) disc are marked; waypoints are cell centers,
        // so planned paths keep at least the plan radius from the threat, and
        // the movement gate enforces a slightly smaller radius (hysteresis) so
        // the bounded clip of cell-adjacent segments can never trip it. Discs
        // smaller than ~1.5 cells break that sagitta bound, so they are marked
        // conservatively by rectangle touch instead.
        //
        // A disc currently containing the agent (it swept over us) is marked
        // soft — traversable at a cost — so an escape plan always exists; all
        // other discs stay hard so a route can never be planned through them.
        for threat in nearby.iter() {
            let containing = distance_sq(position, threat.center) <= threat.radius * threat.radius;
            let tiny = threat.radius < cell * 1.5;
            let min_c = (((threat.center[0] - threat.radius - sensed.origin[0]) / cell).floor()
                as isize)
                .max(window.min_col as isize);
            let max_c = (((threat.center[0] + threat.radius - sensed.origin[0]) / cell).floor()
                as isize)
                .min((window.min_col + window.width - 1) as isize);
            let min_r = (((threat.center[1] - threat.radius - sensed.origin[1]) / cell).floor()
                as isize)
                .max(window.min_row as isize);
            let max_r = (((threat.center[1] + threat.radius - sensed.origin[1]) / cell).floor()
                as isize)
                .min((window.min_row + window.height - 1) as isize);
            for row in min_r..=max_r {
                for col in min_c..=max_c {
                    let rect_min = [
                        sensed.origin[0] + col as f64 * cell,
                        sensed.origin[1] + row as f64 * cell,
                    ];
                    let probe = if tiny {
                        [
                            threat.center[0].clamp(rect_min[0], rect_min[0] + cell),
                            threat.center[1].clamp(rect_min[1], rect_min[1] + cell),
                        ]
                    } else {
                        [rect_min[0] + cell * 0.5, rect_min[1] + cell * 0.5]
                    };
                    if distance_sq(probe, threat.center) <= threat.radius * threat.radius {
                        let index = window.index(col, row);
                        if containing {
                            pen_stamp[index] = soft_generation;
                        } else {
                            mark_stamp[index] = mark_generation;
                        }
                    }
                }
            }
        }

        // Emergency demotion: if every orthogonal step out of the start cell
        // is hard-closed (rare disc/terrain geometry), hard marks become soft
        // for this plan so a way out always exists. The movement gate stays
        // the authority that the agent never moves deeper into a disc.
        let start_index = window.index(acol as isize, arow as isize);
        let hard_open = |col: isize, row: isize| -> bool {
            window.contains(col, row)
                && !sensed.blocked(col, row)
                && mark_stamp[window.index(col, row)] != mark_generation
        };
        let demote_hard = !NEIGHBORS[..4]
            .iter()
            .any(|&(dx, dz, _)| hard_open(acol as isize + dx, arow as isize + dz));
        let open = |col: isize, row: isize| -> bool {
            window.contains(col, row)
                && !sensed.blocked(col, row)
                && (demote_hard || mark_stamp[window.index(col, row)] != mark_generation)
        };

        // Clamp the target into the window and nudge it to the nearest open
        // cell. A corridor point is an optimistic coarse cell center and can
        // sit well inside blocked fine terrain, so the nudge must reach at
        // least across one coarse cell.
        let tcol = (((target_world[0] - sensed.origin[0]) / cell).floor() as isize)
            .clamp(window.min_col as isize, (window.min_col + window.width - 1) as isize);
        let trow = (((target_world[1] - sensed.origin[1]) / cell).floor() as isize)
            .clamp(window.min_row as isize, (window.min_row + window.height - 1) as isize);
        let nudge_reach = (cfg.coarse_factor as isize + 4).max(6);
        let mut target_cell = None;
        'nudge: for radius in 0..=nudge_reach {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    if dx.abs().max(dz.abs()) != radius {
                        continue;
                    }
                    let (col, row) = (tcol + dx, trow + dz);
                    if open(col, row) {
                        target_cell = Some((col, row));
                        break 'nudge;
                    }
                }
            }
        }
        let Some((tcol_open, trow_open)) = target_cell else {
            if std::env::var_os("NAV_DEBUG").is_some() {
                eprintln!("  nudge failed around target {target_world:?}");
            }
            return failed;
        };
        let nudged = (tcol_open, trow_open) != (tcol, trow);
        let target_index = window.index(tcol_open, trow_open);
        let cellf = cell as f32;
        let h_of = |index: usize| -> f32 {
            let col = (window.min_col + index % window.width) as isize;
            let row = (window.min_row + index / window.width) as isize;
            (octile((col - tcol_open) as f64, (row - trow_open) as f64) * cell) as f32
        };

        heap.clear();
        stamp[start_index] = generation;
        g[start_index] = 0.0;
        came[start_index] = u32::MAX;
        heap.push(HeapNode {
            score: h_of(start_index),
            index: start_index as u32,
        });
        let start_h = h_of(start_index);
        let mut best_h = start_h;
        let mut best_index = start_index;
        let mut found = start_index == target_index;
        let mut expansions = 0_usize;
        while let Some(node) = heap.pop() {
            let index = node.index as usize;
            if closed[index] == generation {
                continue;
            }
            closed[index] = generation;
            let h = h_of(index);
            if h < best_h {
                best_h = h;
                best_index = index;
            }
            if index == target_index {
                found = true;
                break;
            }
            expansions += 1;
            if expansions > cfg.local_expansion_cap {
                break;
            }
            let col = (window.min_col + index % window.width) as isize;
            let row = (window.min_row + index / window.width) as isize;
            for (dx, dz, base) in NEIGHBORS {
                let ncol = col + dx;
                let nrow = row + dz;
                if !open(ncol, nrow) {
                    continue;
                }
                if dx != 0 && dz != 0 && (!open(ncol, row) || !open(col, nrow)) {
                    continue;
                }
                let nindex = window.index(ncol, nrow);
                let mut step = base as f32 * cellf;
                if pen_stamp[nindex] == soft_generation
                    || (demote_hard && mark_stamp[nindex] == mark_generation)
                {
                    // Soft threat cells stay traversable but expensive, so the
                    // shortest way out of a disc that swept over us wins.
                    step += 12.0 * cellf;
                }
                if stamp[nindex] != generation {
                    stamp[nindex] = generation;
                    g[nindex] = f32::INFINITY;
                    came[nindex] = u32::MAX;
                }
                let candidate = g[index] + step;
                if candidate < g[nindex] {
                    g[nindex] = candidate;
                    came[nindex] = index as u32;
                    heap.push(HeapNode {
                        score: candidate + h_of(nindex),
                        index: nindex as u32,
                    });
                }
            }
        }

        let end_index = if found {
            target_index
        } else if best_h <= start_h - cellf {
            // Partial progress toward an unreachable target is still useful;
            // anything less means the frontier never left the start cell.
            best_index
        } else {
            if std::env::var_os("NAV_DEBUG").is_some() {
                eprintln!(
                    "  no progress: target {target_world:?} start_h {start_h:.1} best_h {best_h:.1} expansions {expansions}"
                );
            }
            return failed;
        };

        path_cells.clear();
        let mut cursor = end_index;
        loop {
            path_cells.push(cursor as u32);
            if cursor == start_index {
                break;
            }
            let prior = came[cursor];
            if prior == u32::MAX {
                return failed;
            }
            cursor = prior as usize;
        }
        points.clear();
        points.push(position);
        for &windex in path_cells.iter().rev().skip(1) {
            let windex = windex as usize;
            let col = window.min_col + windex % window.width;
            let row = window.min_row + windex / window.width;
            points.push(sensed.center(col, row));
        }
        if found && !nudged {
            let last = *points.last().expect("points always has the start");
            if distance_sq(last, target_world) > 1e-12 {
                points.push(target_world);
            }
        }

        // String pulling: consecutive points are safe by construction, longer
        // jumps must pass the conservative terrain raycast and stay clear of
        // every threat disc — a jump only checked against terrain could cut
        // through cells the A* path deliberately avoided. Only a disc that
        // currently contains the agent is exempt (every jump near the start
        // would otherwise fail); the movement gate still forbids moving deeper
        // into it.
        let jump_is_clear = |from: [f64; 2], to: [f64; 2]| -> bool {
            sensed.segment_is_clear(from, to)
                && nearby.iter().all(|threat| {
                    distance_sq(position, threat.center) <= threat.radius * threat.radius
                        || segment_distance_sq(from, to, threat.center)
                            > threat.radius * threat.radius
                })
        };
        self.local.clear();
        self.local.push(points[0]);
        let mut index = 0;
        while index + 1 < points.len() {
            let mut jump = (index + 40).min(points.len() - 1);
            while jump > index + 1 && !jump_is_clear(points[index], points[jump]) {
                jump -= 1;
            }
            self.local.push(points[jump]);
            index = jump;
        }
        self.local_next = 1;
        LocalOutcome::Planned
    }

    /// Reactive escape while inside one or more threat discs: steer directly
    /// away from the containing discs (deeper discs weigh more), rotating the
    /// heading until the movement gate accepts a step. Recomputed every tick,
    /// so it tracks moving threats with zero plan staleness.
    fn flee(&mut self, shared: &NavShared, sensed: &WalkView, dt: f64) -> f64 {
        let mut away = [0.0_f64, 0.0_f64];
        for threat in &self.scratch.nearby {
            let d_sq = distance_sq(self.position, threat.center);
            if d_sq > threat.radius * threat.radius {
                continue;
            }
            let d = d_sq.sqrt().max(1e-6);
            let depth = (threat.radius - d) / threat.radius;
            let weight = 1.0 + depth * 4.0;
            away[0] += (self.position[0] - threat.center[0]) / d * weight;
            away[1] += (self.position[1] - threat.center[1]) / d * weight;
        }
        let length = away[0].hypot(away[1]);
        let base = if length > 1e-9 {
            [away[0] / length, away[1] / length]
        } else {
            [1.0, 0.0]
        };
        let step = shared.config.speed.max(0.0) * dt;
        // The fan includes the exact tangents (±π/2): sliding along the shell
        // of the pinning disc is always non-inward, so a wall-pinned agent
        // can slide out sideways instead of holding.
        for angle in [
            0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.3,
            -1.3,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
            1.9,
            -1.9,
            2.2,
            -2.2,
        ] {
            let (sin, cos) = f64::sin_cos(angle);
            let direction = [base[0] * cos - base[1] * sin, base[0] * sin + base[1] * cos];
            let next = [
                self.position[0] + direction[0] * step,
                self.position[1] + direction[1] * step,
            ];
            if self.move_allowed(shared, sensed, self.position, next) {
                self.position = next;
                return step;
            }
        }
        0.0
    }

    fn steer(&mut self, shared: &NavShared, sensed: &WalkView, dt: f64) -> f64 {
        let cfg = &shared.config;
        let mut remaining = cfg.speed.max(0.0) * dt;
        let mut moved = 0.0;
        while remaining > 1e-9 && self.local_next < self.local.len() {
            let target = self.local[self.local_next];
            let delta = [target[0] - self.position[0], target[1] - self.position[1]];
            let length = delta[0].hypot(delta[1]);
            if length <= 1e-9 {
                self.local_next += 1;
                continue;
            }
            let step = remaining.min(length);
            let next = [
                self.position[0] + delta[0] / length * step,
                self.position[1] + delta[1] / length * step,
            ];
            if !self.move_allowed(shared, sensed, self.position, next) {
                // The plan was invalidated (streamed obstacle or a threat moved
                // in): hold position and replan on the next tick.
                self.local.clear();
                self.local_next = 0;
                self.replan_in = 0.0;
                break;
            }
            self.position = next;
            moved += step;
            remaining -= step;
            if step >= length - 1e-9 {
                self.local_next += 1;
            }
        }
        moved
    }

    /// Movement gate: every interpolated step must pass the supercover raycast
    /// against the sensed grid and stay outside every threat disc. The gate
    /// enforces half the threat margin less than the planner avoids, so a
    /// legally planned path can never chatter against the gate. Discs that
    /// swept over the agent are judged together: a step is admitted when the
    /// TOTAL penetration depth across all containing discs does not increase,
    /// so a single disc can be escaped but never crossed through, while
    /// overlapping discs cannot wedge the agent by each vetoing the only
    /// direction that escapes the other.
    fn move_allowed(
        &self,
        shared: &NavShared,
        sensed: &WalkView,
        from: [f64; 2],
        to: [f64; 2],
    ) -> bool {
        let escaping_cell = sensed
            .cell_of(from)
            .is_some_and(|(col, row)| sensed.blocked(col as isize, row as isize));
        if !sensed.segment_clear_from(from, to, escaping_cell) {
            return false;
        }
        let slack = shared.config.threat_margin * 0.5;
        let mut depth_from = 0.0;
        let mut depth_to = 0.0;
        for threat in &self.scratch.nearby {
            let gate_radius = (threat.radius - slack).max(0.0);
            let radius_sq = gate_radius * gate_radius;
            let from_distance_sq = distance_sq(from, threat.center);
            if from_distance_sq <= radius_sq {
                depth_from += gate_radius - from_distance_sq.sqrt();
                depth_to += (gate_radius - distance(to, threat.center)).max(0.0);
            } else if segment_distance_sq(from, to, threat.center) <= radius_sq {
                return false;
            }
        }
        depth_to <= depth_from + 1e-9
    }
}

fn nearest_open_coarse(coarse: &CoarseView, cell: (usize, usize)) -> Option<(usize, usize)> {
    for radius in 0..=3_isize {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                if dx.abs().max(dz.abs()) != radius {
                    continue;
                }
                let col = cell.0 as isize + dx;
                let row = cell.1 as isize + dz;
                if coarse.is_open(col, row) {
                    return Some((col as usize, row as usize));
                }
            }
        }
    }
    None
}

fn octile(dc: f64, dr: f64) -> f64 {
    let dc = dc.abs();
    let dr = dr.abs();
    dc.max(dr) + (std::f64::consts::SQRT_2 - 1.0) * dc.min(dr)
}

fn distance_sq(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

fn distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    distance_sq(left, right).sqrt()
}

fn segment_distance_sq(from: [f64; 2], to: [f64; 2], point: [f64; 2]) -> f64 {
    let delta = [to[0] - from[0], to[1] - from[1]];
    let length_sq = delta[0] * delta[0] + delta[1] * delta[1];
    let t = if length_sq <= f64::EPSILON {
        0.0
    } else {
        (((point[0] - from[0]) * delta[0] + (point[1] - from[1]) * delta[1]) / length_sq)
            .clamp(0.0, 1.0)
    };
    let nearest = [from[0] + delta[0] * t, from[1] + delta[1] * t];
    distance_sq(point, nearest)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn view(width: usize, height: usize, blocked: impl Fn(usize, usize) -> bool) -> WalkView {
        let cells: Vec<u8> = (0..width * height)
            .map(|index| u8::from(blocked(index % width, index / width)))
            .collect();
        WalkView::from_blocked(width, height, [0.0, 0.0], 1.0, &cells)
    }

    fn test_config() -> NavConfig {
        NavConfig {
            coarse_factor: 4,
            sense_radius: 12.0,
            speed: 4.0,
            replan_seconds: 0.2,
            max_fail_streak: 24,
            ..NavConfig::default()
        }
    }

    struct RunResult {
        reached: bool,
        status: NavStatus,
        min_threat_distance: f64,
    }

    /// Drive a navigator to `goal`, asserting on EVERY tick that the swept,
    /// interpolated segment never crosses ground-truth collision terrain and
    /// keeps the configured clearance radius from every blocked cell (checked
    /// with exact segment-to-rectangle geometry against the RAW grid).
    /// With `stream`, the agent's known map starts as the shared base view and
    /// ground-truth cells are copied in only within the sensing radius,
    /// simulating obstacles being streamed as the agent approaches.
    fn run(
        shared: &NavShared,
        truth: &WalkView,
        threats: &[Threat],
        start: [f64; 2],
        goal: [f64; 2],
        stream: bool,
        max_ticks: usize,
    ) -> RunResult {
        let radius = shared.config.agent_radius;
        let plan_view = if radius > 0.0 {
            truth.dilated(radius)
        } else {
            truth.clone()
        };
        let mut known = if stream {
            shared.fine.clone()
        } else {
            plan_view.clone()
        };
        let mut navigator = Navigator::new(start);
        navigator.set_goal(goal);
        let dt = 0.05;
        let mut min_threat = f64::INFINITY;
        for _ in 0..max_ticks {
            if stream {
                stream_sense(
                    &mut known,
                    &plan_view,
                    navigator.position(),
                    shared.config.sense_radius,
                );
            }
            let before = navigator.position();
            let status = navigator.tick(shared, &known, threats, dt);
            let after = navigator.position();
            assert!(
                truth.segment_is_clear(before, after),
                "interpolated move {before:?} -> {after:?} crossed collision terrain"
            );
            assert_clearance(truth, before, after, radius);
            for threat in threats {
                min_threat = min_threat.min(distance(after, threat.center));
            }
            match status {
                NavStatus::Arrived => {
                    return RunResult {
                        reached: true,
                        status,
                        min_threat_distance: min_threat,
                    }
                }
                NavStatus::Blocked => {
                    return RunResult {
                        reached: false,
                        status,
                        min_threat_distance: min_threat,
                    }
                }
                _ => {}
            }
        }
        RunResult {
            reached: false,
            status: NavStatus::Moving,
            min_threat_distance: min_threat,
        }
    }

    fn stream_sense(known: &mut WalkView, truth: &WalkView, position: [f64; 2], radius: f64) {
        let cell = known.cell_size;
        let min_col = (((position[0] - radius) / cell).floor() as isize).max(0);
        let max_col = (((position[0] + radius) / cell).floor() as isize)
            .min(known.width as isize - 1);
        let min_row = (((position[1] - radius) / cell).floor() as isize).max(0);
        let max_row = (((position[1] + radius) / cell).floor() as isize)
            .min(known.height as isize - 1);
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                let center = known.center(col as usize, row as usize);
                if distance_sq(center, position) <= radius * radius {
                    known.set_blocked(
                        col as usize,
                        row as usize,
                        truth.blocked(col, row),
                    );
                }
            }
        }
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
        segment_distance_sq(a1, a2, b1)
            .min(segment_distance_sq(a1, a2, b2))
            .min(segment_distance_sq(b1, b2, a1))
            .min(segment_distance_sq(b1, b2, a2))
            .sqrt()
    }

    fn segment_rect_distance(
        from: [f64; 2],
        to: [f64; 2],
        rect_min: [f64; 2],
        rect_max: [f64; 2],
    ) -> f64 {
        let inside = |p: [f64; 2]| {
            p[0] >= rect_min[0] && p[0] <= rect_max[0] && p[1] >= rect_min[1] && p[1] <= rect_max[1]
        };
        if inside(from) || inside(to) {
            return 0.0;
        }
        let corners = [
            [rect_min[0], rect_min[1]],
            [rect_max[0], rect_min[1]],
            [rect_max[0], rect_max[1]],
            [rect_min[0], rect_max[1]],
        ];
        (0..4)
            .map(|index| seg_seg_distance(from, to, corners[index], corners[(index + 1) % 4]))
            .fold(f64::INFINITY, f64::min)
    }

    /// Exact geometric check that a disc of `radius` swept along the segment
    /// never clips any blocked cell of the raw grid.
    fn assert_clearance(truth: &WalkView, from: [f64; 2], to: [f64; 2], radius: f64) {
        if radius <= 0.0 {
            return;
        }
        let cell = truth.cell_size;
        let reach = (radius / cell).ceil() as isize + 1;
        let min_col = ((((from[0].min(to[0]) - truth.origin[0]) / cell).floor() as isize) - reach)
            .max(0);
        let max_col = ((((from[0].max(to[0]) - truth.origin[0]) / cell).floor() as isize) + reach)
            .min(truth.width as isize - 1);
        let min_row = ((((from[1].min(to[1]) - truth.origin[1]) / cell).floor() as isize) - reach)
            .max(0);
        let max_row = ((((from[1].max(to[1]) - truth.origin[1]) / cell).floor() as isize) + reach)
            .min(truth.height as isize - 1);
        for row in min_row..=max_row {
            for col in min_col..=max_col {
                if !truth.blocked(col, row) {
                    continue;
                }
                let rect_min = [
                    truth.origin[0] + col as f64 * cell,
                    truth.origin[1] + row as f64 * cell,
                ];
                let rect_max = [rect_min[0] + cell, rect_min[1] + cell];
                let clearance = segment_rect_distance(from, to, rect_min, rect_max);
                assert!(
                    clearance + 1e-6 >= radius,
                    "clearance violated: segment {from:?} -> {to:?} passes {clearance:.3} from \
                     blocked cell ({col},{row}), need {radius}"
                );
            }
        }
    }

    #[test]
    fn supercover_detects_wall_hit_and_corner_cut() {
        let walled = view(10, 10, |col, row| col == 5 && row != 5);
        assert!(!walled.segment_is_clear([2.5, 2.5], [8.5, 2.5]));
        assert!(walled.segment_is_clear([2.5, 5.5], [8.5, 5.5]));

        let corner = view(10, 10, |col, row| (col, row) == (5, 5) || (col, row) == (4, 4));
        assert!(
            !corner.segment_is_clear([4.5, 5.5], [5.5, 4.5]),
            "exact corner crossing between two diagonal blocks must be conservative"
        );
        assert!(corner.segment_is_clear([2.5, 7.5], [7.5, 7.5]));
        // Leaving the grid is never clear.
        assert!(!corner.segment_is_clear([2.5, 2.5], [12.5, 2.5]));
    }

    #[test]
    fn dilated_view_blocks_cells_near_walls() {
        let base = view(5, 5, |col, row| (col, row) == (2, 2));
        let fat = base.dilated(1.0);
        for row in 1..=3_isize {
            for col in 1..=3_isize {
                assert!(fat.blocked(col, row), "({col},{row}) should be dilated");
            }
        }
        assert!(!fat.blocked(0, 2));
        assert!(!fat.blocked(2, 0));
        assert!(!base.blocked(1, 1), "base view must stay untouched");
    }

    #[test]
    fn maze_steering_reaches_goal_without_crossing_collision() {
        // Thick walls (full coarse columns) with alternating gaps: the coarse
        // view routes the zig-zag, the fine window walks it.
        let truth = view(64, 64, |col, row| {
            ((16..20).contains(&col) && row >= 8)
                || ((32..36).contains(&col) && row < 56)
                || ((48..52).contains(&col) && row >= 8)
        });
        let shared = NavShared::from_view(truth.clone(), test_config());
        let result = run(&shared, &truth, &[], [2.5, 2.5], [61.5, 61.5], false, 6000);
        assert!(result.reached, "status {:?}", result.status);
    }

    #[test]
    fn hidden_obstacles_streamed_in_are_avoided() {
        // The shared/known map is empty; ground truth has small 2x2 blobs that
        // only stream into the known map inside the sensing radius.
        let empty = view(80, 80, |_, _| false);
        let truth = view(80, 80, |col, row| {
            (0..6).any(|i| {
                (0..7).any(|j| {
                    let (bx, bz) = (10 + 13 * i, 7 + 11 * j);
                    (bx..bx + 2).contains(&col) && (bz..bz + 2).contains(&row)
                })
            })
        });
        let shared = NavShared::from_view(empty, test_config());
        let result = run(&shared, &truth, &[], [2.5, 2.5], [76.5, 76.5], true, 6000);
        assert!(result.reached, "status {:?}", result.status);
    }

    #[test]
    fn route_keeps_outside_enemy_awareness() {
        let truth = view(48, 48, |_, _| false);
        let shared = NavShared::from_view(truth.clone(), test_config());
        let threats = [Threat {
            center: [24.0, 24.0],
            radius: 6.0,
        }];
        let result = run(&shared, &truth, &threats, [4.5, 24.5], [44.5, 24.5], false, 4000);
        assert!(result.reached, "status {:?}", result.status);
        assert!(
            result.min_threat_distance > 6.0,
            "entered enemy radius: closest approach {:.3}",
            result.min_threat_distance
        );
    }

    #[test]
    fn blocked_when_every_route_enters_enemy_radius() {
        // A single corridor fully covered by the enemy awareness disc.
        let truth = view(40, 16, |_, row| !(6..10).contains(&row));
        let shared = NavShared::from_view(truth.clone(), test_config());
        let threats = [Threat {
            center: [20.0, 8.0],
            radius: 4.0,
        }];
        let result = run(&shared, &truth, &threats, [4.5, 8.5], [36.5, 8.5], false, 4000);
        assert!(!result.reached);
        assert_eq!(result.status, NavStatus::Blocked);
        assert!(
            result.min_threat_distance > 4.0,
            "entered enemy radius while probing: closest approach {:.3}",
            result.min_threat_distance
        );
    }

    #[test]
    fn optimistic_coarse_gap_is_repaired_by_penalties() {
        // The wall spans coarse column 7 (fine cols 28..32). The pocket at
        // rows 30..36 is open for the wall's first three columns but dead-ends
        // against the last, so the coarse view promises a passage that does
        // not exist even after clearance dilation; the real gap is at rows
        // 55..61. The navigator must penalize the fake gap and reroute.
        let truth = view(64, 64, |col, row| {
            if !(28..32).contains(&col) {
                return false;
            }
            if (55..61).contains(&row) {
                return false;
            }
            if (30..36).contains(&row) {
                return col == 31;
            }
            true
        });
        let shared = NavShared::from_view(truth.clone(), test_config());
        let result = run(&shared, &truth, &[], [10.5, 32.5], [54.5, 32.5], false, 8000);
        assert!(result.reached, "status {:?}", result.status);
    }

    #[test]
    fn wandering_mobs_aggro_is_never_breached_to_080_radius() {
        // Live simulation: mobs wander around home points while the agent
        // walks long crossings between distant waypoints through their field.
        // The lanes between aggro discs open and close as the mobs drift, so
        // discs regularly sweep over the moving agent; the agent must never be
        // found inside 0.80x of any raw aggro radius.
        const AGGRO: f64 = 4.0;
        const MOB_SPEED: f64 = 1.5;
        const HOME_RADIUS: f64 = 6.0;
        let spots = [14_usize, 34, 54, 74];
        let truth = view(96, 96, |col, row| {
            spots.iter().any(|&bx| {
                spots.iter().any(|&bz| {
                    (bx == 14 || bx == 74 || bz == 14 || bz == 74)
                        && (bx..bx + 2).contains(&col)
                        && (bz..bz + 2).contains(&row)
                })
            })
        });
        let shared = NavShared::from_view(truth.clone(), test_config());
        let plan_view = truth.dilated(shared.config.agent_radius);

        struct Mob {
            position: [f64; 2],
            home: [f64; 2],
            direction: [f64; 2],
            retarget: f64,
        }
        let mut rng = 0x2545_F491_4F6C_DD1D_u64;
        let roll = |state: &mut u64| {
            let mut value = *state;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            *state = value;
            value as f64 / u64::MAX as f64
        };
        let mut mobs: Vec<Mob> = [32.0, 48.0, 64.0]
            .iter()
            .flat_map(|&x| {
                [32.0, 48.0, 64.0].map(move |z| Mob {
                    position: [x, z],
                    home: [x, z],
                    direction: [0.0, 0.0],
                    retarget: 0.0,
                })
            })
            .collect();
        let goals = [[89.5, 89.5], [6.5, 89.5], [89.5, 6.5], [6.5, 6.5]];
        let mut goal_index = 0;
        let mut navigator = Navigator::new([6.5, 6.5]);
        navigator.set_goal(goals[goal_index]);

        let dt = 0.05;
        let mut arrivals = 0;
        let mut min_ratio = f64::INFINITY;
        let mut threats = Vec::with_capacity(mobs.len());
        for _ in 0..12_000 {
            for mob in &mut mobs {
                mob.retarget -= dt;
                if mob.retarget <= 0.0 {
                    let angle = roll(&mut rng) * std::f64::consts::TAU;
                    mob.direction = [angle.sin(), angle.cos()];
                    mob.retarget = 0.8 + roll(&mut rng) * 2.0;
                }
                let mut next = [
                    mob.position[0] + mob.direction[0] * MOB_SPEED * dt,
                    mob.position[1] + mob.direction[1] * MOB_SPEED * dt,
                ];
                if distance(next, mob.home) > HOME_RADIUS {
                    let home = [mob.home[0] - mob.position[0], mob.home[1] - mob.position[1]];
                    let length = home[0].hypot(home[1]).max(1e-9);
                    mob.direction = [home[0] / length, home[1] / length];
                    next = [
                        mob.position[0] + mob.direction[0] * MOB_SPEED * dt,
                        mob.position[1] + mob.direction[1] * MOB_SPEED * dt,
                    ];
                }
                mob.position = next;
            }
            threats.clear();
            threats.extend(mobs.iter().map(|mob| Threat {
                center: mob.position,
                radius: AGGRO,
            }));

            let before = navigator.position();
            let status = navigator.tick(&shared, &plan_view, &threats, dt);
            let after = navigator.position();
            assert!(
                truth.segment_is_clear(before, after),
                "interpolated move {before:?} -> {after:?} crossed collision terrain"
            );
            assert_clearance(&truth, before, after, shared.config.agent_radius);
            for mob in &mobs {
                let separation = distance(after, mob.position);
                min_ratio = min_ratio.min(separation / AGGRO);
                assert!(
                    separation > AGGRO * 0.8,
                    "agent breached aggro to {separation:.2} (0.8x = {:.2}) at {after:?}, \
                     mob at {:?}",
                    AGGRO * 0.8,
                    mob.position
                );
            }
            match status {
                NavStatus::Arrived | NavStatus::Blocked | NavStatus::Idle => {
                    if status == NavStatus::Arrived {
                        arrivals += 1;
                    }
                    goal_index = (goal_index + 1) % goals.len();
                    navigator.set_goal(goals[goal_index]);
                }
                NavStatus::Moving => {}
            }
        }
        println!("arrivals {arrivals}, min separation ratio {min_ratio:.3}");
        assert!(
            arrivals >= 2,
            "expected repeated field crossings, got {arrivals} arrivals"
        );
    }

    #[test]
    fn short_range_goals_are_reached() {
        // Goals only a fraction of a cell away (same cell, orthogonal and
        // diagonal neighbours) must still be walked to, from positions on and
        // off cell centers.
        let truth = view(32, 32, |_, _| false);
        let shared = NavShared::from_view(truth.clone(), test_config());
        let starts = [[10.5, 10.5], [10.3, 10.7], [10.999, 10.001]];
        let offsets = [
            [0.5, 0.0],
            [-0.5, 0.0],
            [0.0, 0.5],
            [0.4, 0.4],
            [-0.6, 0.8],
            [1.2, 0.0],
            [1.4, 1.4],
        ];
        for start in starts {
            for offset in offsets {
                let goal = [start[0] + offset[0], start[1] + offset[1]];
                let mut navigator = Navigator::new(start);
                navigator.set_goal(goal);
                let mut status = NavStatus::Moving;
                for _ in 0..200 {
                    status = navigator.tick(&shared, &shared.fine, &[], 0.05);
                    if status != NavStatus::Moving {
                        break;
                    }
                }
                assert_eq!(
                    status,
                    NavStatus::Arrived,
                    "goal {goal:?} from {start:?} ended {status:?} at {:?}",
                    navigator.position()
                );
                assert!(
                    distance(navigator.position(), goal) <= 0.36,
                    "stopped {:.3} away from {goal:?}",
                    distance(navigator.position(), goal)
                );
            }
        }
    }

    #[test]
    fn illegal_goals_snap_to_the_nearest_legal_point() {
        // A goal inside the wall's clearance dilation: walk to the nearest
        // open cell instead of refusing to move.
        let walled = view(32, 32, |col, _| col == 20);
        let shared = NavShared::from_view(walled.clone(), test_config());
        let mut navigator = Navigator::new([15.5, 15.5]);
        navigator.set_goal([19.9, 15.5]);
        let mut status = NavStatus::Moving;
        for _ in 0..400 {
            status = navigator.tick(&shared, &shared.fine, &[], 0.05);
            if status != NavStatus::Moving {
                break;
            }
        }
        assert_eq!(status, NavStatus::Arrived, "at {:?}", navigator.position());
        assert!(
            distance(navigator.position(), [19.9, 15.5]) <= 2.0,
            "stopped {:?}, too far from the ordered point",
            navigator.position()
        );

        // A goal inside an enemy shell: stop at the closest safe point
        // outside it instead of refusing (or entering).
        let open = view(32, 32, |_, _| false);
        let shared = NavShared::from_view(open.clone(), test_config());
        let threats = [Threat {
            center: [20.5, 15.5],
            radius: 4.0,
        }];
        let mut navigator = Navigator::new([10.5, 15.5]);
        navigator.set_goal([20.0, 15.5]);
        let mut status = NavStatus::Moving;
        for _ in 0..400 {
            status = navigator.tick(&shared, &shared.fine, &threats, 0.05);
            if status != NavStatus::Moving {
                break;
            }
            let separation = distance(navigator.position(), threats[0].center);
            assert!(separation > 4.0, "entered aggro radius: {separation:.2}");
        }
        assert_eq!(status, NavStatus::Arrived, "at {:?}", navigator.position());
        assert!(
            distance(navigator.position(), threats[0].center) > 5.2,
            "stopped inside the threat shell at {:?}",
            navigator.position()
        );
    }

    #[test]
    fn narrow_corridor_keeps_clearance_on_interpolation() {
        // A four-cell corridor leaves exactly a two-cell strip after the
        // 0.75-unit clearance dilation; the exact per-tick geometry check in
        // `run` proves the swept disc never comes closer than the clearance.
        let truth = view(40, 12, |_, row| !(4..8).contains(&row));
        let shared = NavShared::from_view(truth.clone(), test_config());
        let result = run(&shared, &truth, &[], [2.5, 5.5], [37.5, 6.5], false, 4000);
        assert!(result.reached, "status {:?}", result.status);
    }

    #[test]
    fn many_navigators_smoke() {
        let truth = view(160, 160, |col, row| {
            col % 16 >= 8 && col % 16 < 11 && row % 16 >= 8 && row % 16 < 11
        });
        let shared = NavShared::from_view(truth.clone(), test_config());
        let threats: Vec<Threat> = (0..40)
            .map(|index| Threat {
                center: [
                    4.5 + (index * 37 % 150) as f64,
                    4.5 + (index * 53 % 150) as f64,
                ],
                radius: 3.0,
            })
            .collect();
        let count = 100;
        let ticks = 60;
        let mut navigators: Vec<Navigator> = (0..count)
            .map(|index| {
                let start = [2.5 + (index * 29 % 150) as f64, 2.5 + (index * 41 % 150) as f64];
                let goal = [2.5 + (index * 59 % 150) as f64, 2.5 + (index * 31 % 150) as f64];
                let mut navigator = Navigator::new(start);
                navigator.set_goal(goal);
                navigator
            })
            .collect();
        let clock = Instant::now();
        for _ in 0..ticks {
            for navigator in &mut navigators {
                navigator.tick(&shared, &truth, &threats, 0.05);
            }
        }
        let elapsed = clock.elapsed();
        println!(
            "{count} navigators x {ticks} ticks: {elapsed:?} total, {:.1}us per agent-tick",
            elapsed.as_secs_f64() * 1e6 / (count * ticks) as f64
        );
    }

    #[test]
    #[ignore = "release-mode capacity benchmark: cargo test --release -- --ignored"]
    fn perf_two_thousand_navigators() {
        let truth = view(512, 512, |col, row| {
            col % 24 >= 10 && col % 24 < 14 && row % 24 >= 10 && row % 24 < 14
        });
        let shared = NavShared::from_view(truth.clone(), NavConfig::default());
        let threats: Vec<Threat> = (0..64)
            .map(|index| Threat {
                center: [
                    6.5 + (index * 67 % 500) as f64,
                    6.5 + (index * 101 % 500) as f64,
                ],
                radius: 5.0,
            })
            .collect();
        let count = 2000;
        let ticks = 100;
        let mut navigators: Vec<Navigator> = (0..count)
            .map(|index| {
                let start = [2.5 + (index * 29 % 500) as f64, 2.5 + (index * 41 % 500) as f64];
                let goal = [2.5 + (index * 59 % 500) as f64, 2.5 + (index * 31 % 500) as f64];
                let mut navigator = Navigator::new(start);
                navigator.set_goal(goal);
                navigator
            })
            .collect();
        let clock = Instant::now();
        for _ in 0..ticks {
            for navigator in &mut navigators {
                navigator.tick(&shared, &truth, &threats, 0.05);
            }
        }
        let elapsed = clock.elapsed();
        println!(
            "{count} navigators x {ticks} ticks: {elapsed:?} total, {:.2}us per agent-tick, {:.1}ms per simulated tick",
            elapsed.as_secs_f64() * 1e6 / (count * ticks) as f64,
            elapsed.as_secs_f64() * 1e3 / ticks as f64
        );
    }
}
