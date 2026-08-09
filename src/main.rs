use std::error::Error;
use std::path::PathBuf;

use clap::Parser;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, PointerButton, Pos2, Rect, Sense, Shape, Stroke, Vec2,
};
use nav_scene::{
    find_path, CellPrimitive, Collider, Entity, NavConfig, NavShared, NavStatus, Navigator,
    PathOptions, Scene, SceneStore, Threat,
};

mod population;
mod wander;

use population::{synthetically_populate_actors, PopulationReport, SENSING_RADIUS};
use wander::WanderSimulation;

#[derive(Parser)]
#[command(about = "Game-neutral navigation gym")]
struct Args {
    #[arg(default_value = "data/ravencairn.sqlite")]
    database: PathBuf,
    /// Load and validate the transformed scenario without opening a window.
    #[arg(long)]
    validate_only: bool,
}

struct NavApp {
    scene: Scene,
    population: PopulationReport,
    wander: WanderSimulation,
    database: PathBuf,
    center: [f64; 2],
    zoom: f32,
    needs_fit: bool,
    show_grid: bool,
    show_resources: bool,
    show_actors: bool,
    show_colliders: bool,
    show_ranges: bool,
    show_labels: bool,
    include_dynamic_colliders: bool,
    clearance: f64,
    start: Option<[f64; 2]>,
    goal: Option<[f64; 2]>,
    path: Option<Vec<[f64; 2]>>,
    route_status: String,
    hover: Option<[f64; 2]>,
    nav_shared: NavShared,
    navigator: Navigator,
    hta_enabled: bool,
}

impl NavApp {
    fn new(mut scene: Scene, database: PathBuf) -> Self {
        let bounds = scene.grid.bounds();
        let population = synthetically_populate_actors(&mut scene, SENSING_RADIUS);
        let wander = WanderSimulation::from_scene(&scene);
        let agent = scene
            .entities
            .iter()
            .find(|entity| entity.category == "agent")
            .map(|entity| entity.position);
        let nav_shared = NavShared::from_grid(&scene.grid, NavConfig::default());
        let navigator = Navigator::new(agent.unwrap_or([
            (bounds[0][0] + bounds[1][0]) * 0.5,
            (bounds[0][1] + bounds[1][1]) * 0.5,
        ]));
        Self {
            nav_shared,
            navigator,
            hta_enabled: true,
            center: agent.unwrap_or([
                (bounds[0][0] + bounds[1][0]) * 0.5,
                (bounds[0][1] + bounds[1][1]) * 0.5,
            ]),
            scene,
            population,
            wander,
            database,
            zoom: 1.0,
            needs_fit: true,
            show_grid: true,
            show_resources: true,
            show_actors: true,
            show_colliders: true,
            show_ranges: true,
            show_labels: true,
            include_dynamic_colliders: true,
            clearance: 0.0,
            start: agent,
            goal: None,
            path: None,
            route_status: "left click: start · right click: goal".to_owned(),
            hover: None,
        }
    }

    fn fit(&mut self, size: Vec2) {
        let bounds = self.scene.grid.bounds();
        self.center = [
            (bounds[0][0] + bounds[1][0]) * 0.5,
            (bounds[0][1] + bounds[1][1]) * 0.5,
        ];
        let world_width = (bounds[1][0] - bounds[0][0]).max(1.0) as f32;
        let world_height = (bounds[1][1] - bounds[0][1]).max(1.0) as f32;
        self.zoom = (size.x / world_width)
            .min(size.y / world_height)
            .mul_add(0.94, 0.0)
            .clamp(0.08, 80.0);
    }

    fn tick_navigator(&mut self, ctx: &egui::Context) {
        if !self.hta_enabled || self.navigator.status() != NavStatus::Moving {
            return;
        }
        let dt = f64::from(ctx.input(|input| input.stable_dt)).min(0.1);
        let threats = collect_threats(&self.scene);
        let status = self
            .navigator
            .tick(&self.nav_shared, &self.nav_shared.fine, &threats, dt);
        let position = self.navigator.position();
        if let Some(agent) = self
            .scene
            .entities
            .iter_mut()
            .find(|entity| entity.category == "agent")
        {
            agent.position = position;
        }
        self.route_status = match status {
            NavStatus::Moving => "hta*: moving".to_owned(),
            NavStatus::Arrived => "hta*: arrived".to_owned(),
            NavStatus::Blocked => "hta*: blocked (no route outside enemy radii)".to_owned(),
            NavStatus::Idle => "hta*: idle".to_owned(),
        };
        ctx.request_repaint();
    }

    fn calculate_path(&mut self) {
        let (Some(start), Some(goal)) = (self.start, self.goal) else {
            self.path = None;
            self.route_status = "select both a start and a goal".to_owned();
            return;
        };
        self.path = find_path(
            &self.scene,
            start,
            goal,
            PathOptions {
                include_dynamic_colliders: self.include_dynamic_colliders,
                clearance: self.clearance,
            },
        );
        self.route_status = self.path.as_ref().map_or_else(
            || "no route".to_owned(),
            |path| format!("route: {} cells", path.len()),
        );
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("scene-controls")
            .resizable(false)
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading(&self.scene.metadata.label);
                ui.small(format!("scene: {}", self.scene.metadata.scene_id));
                ui.small(format!(
                    "source revision: {}",
                    self.scene
                        .metadata
                        .source_revision
                        .as_deref()
                        .unwrap_or("—")
                ));
                ui.small(format!("database: {}", self.database.display()));
                ui.separator();
                ui.label(format!(
                    "{} × {} cells · {:.2} units/cell",
                    self.scene.grid.width, self.scene.grid.height, self.scene.grid.cell_size
                ));
                ui.label(format!(
                    "{} entities · {} prefab definitions · {} placements",
                    self.scene.entities.len(),
                    self.scene.prefab_collisions.len(),
                    self.scene.placements.len()
                ));
                let resource_count = self
                    .scene
                    .entities
                    .iter()
                    .filter(|entity| entity.category == "resource")
                    .count();
                let actor_count = self
                    .scene
                    .entities
                    .iter()
                    .filter(|entity| entity.category == "actor")
                    .count();
                ui.label(format!("{resource_count} resources · {actor_count} actors"));
                ui.small(format!(
                    "{} sensed within {:.0} · {} synthetic · {:.4}/unit²",
                    self.population.observed_actors,
                    self.population.sensing_radius,
                    self.population.synthetic_actors,
                    self.population.density_per_square_unit,
                ));
                ui.separator();
                ui.heading("Actor simulation");
                ui.checkbox(&mut self.wander.enabled, "Wander from spawn centers");
                ui.add(
                    egui::Slider::new(&mut self.wander.radius, 1.0..=20.0)
                        .step_by(0.5)
                        .text("home radius"),
                );
                ui.add(
                    egui::Slider::new(&mut self.wander.speed, 0.25..=5.0)
                        .step_by(0.25)
                        .text("speed"),
                );
                if ui.button("Reset actors to spawn").clicked() {
                    self.wander.reset(&mut self.scene);
                }
                ui.separator();
                ui.checkbox(&mut self.show_grid, "Map collision geometry");
                ui.checkbox(&mut self.show_resources, "Resources");
                ui.checkbox(&mut self.show_actors, "Actors");
                ui.checkbox(&mut self.show_colliders, "Entity colliders");
                ui.checkbox(&mut self.show_ranges, "Entity ranges");
                ui.checkbox(&mut self.show_labels, "Labels");
                ui.separator();
                ui.heading("Pathfinding");
                ui.checkbox(&mut self.hta_enabled, "HTA* steering (avoid enemy aggro)");
                if self.hta_enabled {
                    ui.add(
                        egui::Slider::new(&mut self.nav_shared.config.speed, 0.5..=10.0)
                            .step_by(0.25)
                            .text("agent speed"),
                    );
                    if ui.button("Stop agent").clicked() {
                        self.navigator.stop();
                        self.route_status = "hta*: idle".to_owned();
                    }
                    ui.small("Left click teleports the agent · right click sets its goal");
                }
                ui.checkbox(
                    &mut self.include_dynamic_colliders,
                    "Avoid entity colliders",
                );
                ui.add(
                    egui::Slider::new(&mut self.clearance, 0.0..=3.0)
                        .step_by(0.25)
                        .text("clearance"),
                );
                if ui.button("Use agent as start").clicked() {
                    self.start = self
                        .scene
                        .entities
                        .iter()
                        .find(|entity| entity.category == "agent")
                        .map(|entity| entity.position);
                    self.calculate_path();
                }
                if ui.button("Calculate route").clicked() {
                    self.calculate_path();
                }
                if ui.button("Clear route").clicked() {
                    self.goal = None;
                    self.path = None;
                    self.route_status = "left click: start · right click: goal".to_owned();
                }
                ui.label(&self.route_status);
                ui.separator();
                if ui.button("Fit scene").clicked() {
                    self.needs_fit = true;
                }
                if let Some(world) = self.hover {
                    ui.monospace(format!("x {:>8.2}  z {:>8.2}", world[0], world[1]));
                }
                ui.small("Drag to pan · wheel to zoom");
                ui.small("Left click sets start · right click sets goal");
            });
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        painter.rect_filled(rect, 0.0, Color32::from_rgb(13, 16, 20));
        if self.needs_fit && rect.width() > 1.0 && rect.height() > 1.0 {
            self.fit(rect.size());
            self.needs_fit = false;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                let old_zoom = self.zoom;
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.04, 100.0);
                if let Some(pointer) = response.hover_pos() {
                    let before = screen_to_world(pointer, rect, self.center, old_zoom);
                    let after = screen_to_world(pointer, rect, self.center, self.zoom);
                    self.center[0] += before[0] - after[0];
                    self.center[1] += before[1] - after[1];
                }
            }
            self.hover = response
                .hover_pos()
                .map(|position| screen_to_world(position, rect, self.center, self.zoom));
        }
        if response.dragged_by(PointerButton::Primary) || response.dragged_by(PointerButton::Middle)
        {
            let delta = ui.input(|input| input.pointer.delta());
            self.center[0] -= f64::from(delta.x / self.zoom);
            self.center[1] += f64::from(delta.y / self.zoom);
        }
        if response.clicked_by(PointerButton::Primary) {
            if let Some(position) = response.interact_pointer_pos() {
                let world = screen_to_world(position, rect, self.center, self.zoom);
                if self.hta_enabled {
                    self.navigator.set_position(world);
                    if let Some(agent) = self
                        .scene
                        .entities
                        .iter_mut()
                        .find(|entity| entity.category == "agent")
                    {
                        agent.position = world;
                    }
                } else {
                    self.start = Some(world);
                    self.calculate_path();
                }
            }
        }
        if response.clicked_by(PointerButton::Secondary) {
            if let Some(position) = response.interact_pointer_pos() {
                let world = screen_to_world(position, rect, self.center, self.zoom);
                if self.hta_enabled {
                    self.navigator.set_goal(world);
                    self.route_status = "hta*: moving".to_owned();
                } else {
                    self.goal = Some(world);
                    self.calculate_path();
                }
            }
        }

        paint_coordinate_grid(&painter, rect, self.center, self.zoom);
        if self.show_grid {
            paint_nav_grid(&painter, rect, self.center, self.zoom, &self.scene);
        }
        for entity in &self.scene.entities {
            if (entity.category == "resource" && !self.show_resources)
                || (entity.category == "actor" && !self.show_actors)
            {
                continue;
            }
            paint_entity(
                &painter,
                rect,
                self.center,
                self.zoom,
                entity,
                self.show_colliders,
                self.show_ranges,
                self.show_labels,
            );
        }
        if let Some(path) = &self.path {
            let points = path
                .iter()
                .map(|&point| world_to_screen(point, rect, self.center, self.zoom))
                .collect::<Vec<_>>();
            painter.add(Shape::line(
                points,
                Stroke::new(2.5, Color32::from_rgb(90, 220, 255)),
            ));
        }
        if self.hta_enabled {
            let corridor = self.navigator.corridor();
            if corridor.len() > 1 {
                let points = corridor
                    .iter()
                    .map(|&point| world_to_screen(point, rect, self.center, self.zoom))
                    .collect::<Vec<_>>();
                painter.add(Shape::line(
                    points,
                    Stroke::new(1.5, Color32::from_rgba_unmultiplied(180, 180, 200, 110)),
                ));
            }
            let remaining = self.navigator.local_remaining();
            if !remaining.is_empty() {
                let mut points = vec![world_to_screen(
                    self.navigator.position(),
                    rect,
                    self.center,
                    self.zoom,
                )];
                points.extend(
                    remaining
                        .iter()
                        .map(|&point| world_to_screen(point, rect, self.center, self.zoom)),
                );
                painter.add(Shape::line(
                    points,
                    Stroke::new(2.5, Color32::from_rgb(120, 255, 190)),
                ));
            }
            if let Some(goal) = self.navigator.goal() {
                painter.circle_stroke(
                    world_to_screen(goal, rect, self.center, self.zoom),
                    7.0,
                    Stroke::new(2.5, Color32::from_rgb(120, 255, 190)),
                );
            }
        }
        if let Some(start) = self.start {
            painter.circle_filled(
                world_to_screen(start, rect, self.center, self.zoom),
                5.0,
                Color32::from_rgb(80, 230, 170),
            );
        }
        if let Some(goal) = self.goal {
            let point = world_to_screen(goal, rect, self.center, self.zoom);
            painter.circle_stroke(
                point,
                7.0,
                Stroke::new(2.5, Color32::from_rgb(255, 210, 70)),
            );
        }
    }
}

impl eframe::App for NavApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.wander.enabled && self.wander.actor_count() != 0 {
            let dt = ui.ctx().input(|input| input.stable_dt);
            self.wander.update(&mut self.scene, f64::from(dt));
            ui.ctx().request_repaint();
        }
        self.tick_navigator(ui.ctx());
        self.side_panel(ui);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.canvas(ui));
    }
}

fn collect_threats(scene: &Scene) -> Vec<Threat> {
    scene
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
        .collect()
}

fn world_to_screen(world: [f64; 2], rect: Rect, center: [f64; 2], zoom: f32) -> Pos2 {
    Pos2::new(
        rect.center().x + (world[0] - center[0]) as f32 * zoom,
        rect.center().y - (world[1] - center[1]) as f32 * zoom,
    )
}

fn screen_to_world(screen: Pos2, rect: Rect, center: [f64; 2], zoom: f32) -> [f64; 2] {
    [
        center[0] + f64::from((screen.x - rect.center().x) / zoom),
        center[1] - f64::from((screen.y - rect.center().y) / zoom),
    ]
}

fn paint_nav_grid(painter: &Painter, rect: Rect, center: [f64; 2], zoom: f32, scene: &Scene) {
    let grid = &scene.grid;
    let lower = screen_to_world(rect.left_bottom(), rect, center, zoom);
    let upper = screen_to_world(rect.right_top(), rect, center, zoom);
    let min_col = (((lower[0] - grid.origin[0]) / grid.cell_size).floor() as isize)
        .clamp(0, grid.width as isize - 1) as usize;
    let max_col = (((upper[0] - grid.origin[0]) / grid.cell_size).ceil() as isize)
        .clamp(0, grid.width as isize - 1) as usize;
    let min_row = (((lower[1] - grid.origin[1]) / grid.cell_size).floor() as isize)
        .clamp(0, grid.height as isize - 1) as usize;
    let max_row = (((upper[1] - grid.origin[1]) / grid.cell_size).ceil() as isize)
        .clamp(0, grid.height as isize - 1) as usize;
    let fill = Color32::from_rgba_unmultiplied(235, 82, 74, 92);
    if zoom < 1.15 {
        for row in min_row..=max_row {
            let mut run_start = None;
            for col in min_col..=max_col + 1 {
                let blocked = col <= max_col && grid.blocked[grid.index(col, row)] != 0;
                match (run_start, blocked) {
                    (None, true) => run_start = Some(col),
                    (Some(start), false) => {
                        let min = [
                            grid.origin[0] + start as f64 * grid.cell_size,
                            grid.origin[1] + row as f64 * grid.cell_size,
                        ];
                        let max = [
                            grid.origin[0] + col as f64 * grid.cell_size,
                            min[1] + grid.cell_size,
                        ];
                        painter.rect_filled(
                            Rect::from_two_pos(
                                world_to_screen([min[0], max[1]], rect, center, zoom),
                                world_to_screen([max[0], min[1]], rect, center, zoom),
                            ),
                            0.0,
                            fill,
                        );
                        run_start = None;
                    }
                    _ => {}
                }
            }
        }
        return;
    }
    let stroke = Stroke::new(
        (zoom * 0.08).clamp(0.0, 0.7),
        Color32::from_rgb(232, 104, 94),
    );
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            let index = grid.index(col, row);
            if grid.blocked[index] == 0 {
                continue;
            }
            let min = [
                grid.origin[0] + col as f64 * grid.cell_size,
                grid.origin[1] + row as f64 * grid.cell_size,
            ];
            let max = [min[0] + grid.cell_size, min[1] + grid.cell_size];
            let cell_rect = Rect::from_two_pos(
                world_to_screen([min[0], max[1]], rect, center, zoom),
                world_to_screen([max[0], min[1]], rect, center, zoom),
            );
            let primitive = grid
                .shapes
                .get(&grid.shape_ids[index])
                .unwrap_or(&CellPrimitive::Full);
            paint_cell_primitive(painter, cell_rect, primitive, fill, stroke);
        }
    }
}

fn paint_cell_primitive(
    painter: &Painter,
    rect: Rect,
    primitive: &CellPrimitive,
    fill: Color32,
    stroke: Stroke,
) {
    let local = |point: [f64; 2]| {
        Pos2::new(
            rect.left() + point[0] as f32 * rect.width(),
            rect.bottom() - point[1] as f32 * rect.height(),
        )
    };
    match primitive {
        CellPrimitive::Full => {
            painter.rect_filled(rect, 0.0, fill);
        }
        CellPrimitive::Disc { radius } => {
            painter.circle(rect.center(), *radius as f32 * rect.width(), fill, stroke);
        }
        CellPrimitive::Polygon { points } => {
            painter.add(Shape::convex_polygon(
                points.iter().copied().map(local).collect(),
                fill,
                stroke,
            ));
        }
        CellPrimitive::Edge { from, to } => {
            painter.line_segment(
                [local(*from), local(*to)],
                Stroke::new(stroke.width.max(0.7), stroke.color),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_entity(
    painter: &Painter,
    rect: Rect,
    center: [f64; 2],
    zoom: f32,
    entity: &Entity,
    show_collider: bool,
    show_ranges: bool,
    show_label: bool,
) {
    let point = world_to_screen(entity.position, rect, center, zoom);
    let color = match entity.category.as_str() {
        "agent" => Color32::from_rgb(90, 175, 255),
        "resource" => Color32::from_rgb(85, 215, 115),
        "actor" => Color32::from_rgb(245, 145, 72),
        _ => Color32::from_rgb(205, 205, 215),
    };
    if show_ranges {
        for range in &entity.ranges {
            if range.radius <= 0.0 {
                continue;
            }
            painter.circle_stroke(
                point,
                (range.radius as f32 * zoom).max(1.0),
                Stroke::new(
                    1.0,
                    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 125),
                ),
            );
        }
    }
    if show_collider {
        if let Some(collider) = &entity.collider {
            match collider {
                Collider::Circle { radius } => {
                    painter.circle(
                        point,
                        (*radius as f32 * zoom).max(2.0),
                        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 45),
                        Stroke::new(1.5, color),
                    );
                }
                Collider::Box { half_extents } => {
                    let min = [
                        entity.position[0] - half_extents[0],
                        entity.position[1] - half_extents[1],
                    ];
                    let max = [
                        entity.position[0] + half_extents[0],
                        entity.position[1] + half_extents[1],
                    ];
                    let bounds = Rect::from_two_pos(
                        world_to_screen([min[0], max[1]], rect, center, zoom),
                        world_to_screen([max[0], min[1]], rect, center, zoom),
                    );
                    painter.rect(
                        bounds,
                        0.0,
                        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 45),
                        Stroke::new(1.5, color),
                        egui::StrokeKind::Inside,
                    );
                }
            }
        }
    }
    painter.circle_filled(point, 3.5, color);
    if let Some(heading) = entity.heading_degrees {
        let radians = heading.to_radians();
        let end = [
            entity.position[0] + radians.sin() * 1.5,
            entity.position[1] + radians.cos() * 1.5,
        ];
        painter.line_segment(
            [point, world_to_screen(end, rect, center, zoom)],
            Stroke::new(1.5, color),
        );
    }
    let label_zoom = if entity
        .attributes
        .get("synthetic")
        .is_some_and(|value| value == "true")
    {
        4.0
    } else {
        0.45
    };
    if show_label && zoom >= label_zoom {
        painter.text(
            point + Vec2::new(5.0, -5.0),
            Align2::LEFT_BOTTOM,
            &entity.label,
            FontId::monospace(10.0),
            color,
        );
    }
}

fn paint_coordinate_grid(painter: &Painter, rect: Rect, center: [f64; 2], zoom: f32) {
    let target_world_step = 72.0 / zoom.max(0.001);
    let exponent = target_world_step.log10().floor();
    let base = 10f32.powf(exponent);
    let scaled = target_world_step / base;
    let step = base
        * if scaled <= 2.0 {
            2.0
        } else if scaled <= 5.0 {
            5.0
        } else {
            10.0
        };
    let min = screen_to_world(rect.left_bottom(), rect, center, zoom);
    let max = screen_to_world(rect.right_top(), rect, center, zoom);
    let first_x = (min[0] / f64::from(step)).floor() * f64::from(step);
    let first_z = (min[1] / f64::from(step)).floor() * f64::from(step);
    let lines_x = (((max[0] - first_x) / f64::from(step)).ceil() as usize).min(256);
    let lines_z = (((max[1] - first_z) / f64::from(step)).ceil() as usize).min(256);
    for index in 0..=lines_x {
        let x = first_x + index as f64 * f64::from(step);
        let screen = world_to_screen([x, 0.0], rect, center, zoom).x;
        painter.line_segment(
            [
                Pos2::new(screen, rect.top()),
                Pos2::new(screen, rect.bottom()),
            ],
            Stroke::new(1.0, Color32::from_gray(37)),
        );
    }
    for index in 0..=lines_z {
        let z = first_z + index as f64 * f64::from(step);
        let screen = world_to_screen([0.0, z], rect, center, zoom).y;
        painter.line_segment(
            [
                Pos2::new(rect.left(), screen),
                Pos2::new(rect.right(), screen),
            ],
            Stroke::new(1.0, Color32::from_gray(37)),
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let scene = SceneStore::load(&args.database)?;
    if args.validate_only {
        println!(
            "valid scene {:?}: {}x{} grid, {} entities, {} prefab definitions, {} placements",
            scene.metadata.scene_id,
            scene.grid.width,
            scene.grid.height,
            scene.entities.len(),
            scene.prefab_collisions.len(),
            scene.placements.len()
        );
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([780.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Navigation scene workbench",
        options,
        Box::new(move |_context| Ok(Box::new(NavApp::new(scene, args.database)))),
    )?;
    Ok(())
}
