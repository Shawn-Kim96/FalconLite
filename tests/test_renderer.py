"""Stage 2 renderer tests."""

from falconlite.env import PhysicsConfig, Renderer, RendererConfig, RocketAction, RocketState


def make_renderer() -> Renderer:
    physics_config = PhysicsConfig(
        world_x_limit=100,
        world_y_limit=100,
    )
    render_config = RendererConfig(
        width=500,
        height=400,
        margin=50,
        fps=60,
        rocket_width_px=12,
        rocket_height_px=36,
        pad_width_m=20,
        velocity_scale=2,
    )
    return Renderer(physics_config, render_config)


def test_world_origin_maps_to_screen_center_when_camera_at_origin(monkeypatch) -> None:
    """With the camera-following renderer, before any render() call the camera
    is at the world origin, so (0, 0) maps to the screen center."""

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    renderer = make_renderer()

    try:
        assert renderer.world_to_screen(0, 0) == (250, 200)
    finally:
        renderer.close()


def test_render_accepts_state_action_and_info(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    renderer = make_renderer()
    state = RocketState(x=0, y=50, vx=1, vy=-2, theta=0.1, omega=0.2, fuel=0.75)
    action = RocketAction(thrust=10, gimbal_angle=0.1)

    try:
        renderer.render(
            state,
            action=action,
            info={"done_reason": "running"},
            step=1,
            reward=None,
        )
        assert not renderer.closed
    finally:
        renderer.close()
