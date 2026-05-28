"""Stage 0 import smoke tests."""


def test_core_imports() -> None:
    import falconlite
    from falconlite.controllers import RandomController
    from falconlite.env import RocketState
    from falconlite.utils.config import load_config

    assert falconlite.__version__ == "0.1.0"
    assert RandomController(seed=0).select_action().shape == (2,)
    assert RocketState(x=0, y=1, vx=0, vy=0, theta=0, omega=0, fuel=1).y == 1
    assert load_config()["project"]["name"] == "FalconLite"
