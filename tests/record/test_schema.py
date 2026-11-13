from teleop.config import CameraConfig
from teleop.record.schema import action_names, build_features, state_names


def test_build_features_bimanual_order() -> None:
    state = state_names({"left": ["a"], "right": ["b"]})
    action = action_names({"left": ["a"], "right": ["b"]})
    assert state[:2] == ["left/a", "right/b"]
    assert len(state) == 2 + 14
    feats = build_features(state, action, [CameraConfig()])
    img = feats["observation.images.front"]
    assert img["shape"] == (224, 224, 3)
    assert img["dtype"] == "video"
    assert feats["observation.state"]["shape"] == (16,)
