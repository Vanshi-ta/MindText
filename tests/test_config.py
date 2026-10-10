import pytest

from mindtext.config import Config
from mindtext.errors import ConfigError

REPO_CONFIGS = ["configs/default.yaml", "configs/acoustic.yaml", "configs/reddit_replication.yaml"]


@pytest.mark.parametrize("path", REPO_CONFIGS)
def test_shipped_configs_are_valid(path):
    cfg = Config.from_yaml(path)
    assert cfg.models and cfg.config_hash()


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        Config.from_dict({"data": {"nonsense": 1}, "models": [{"name": "a", "type": "majority"}]})


def test_unknown_model_type_rejected():
    with pytest.raises(ConfigError, match="unknown type"):
        Config.from_dict({"models": [{"name": "a", "type": "magic"}]})


def test_duplicate_model_names_rejected():
    with pytest.raises(ConfigError, match="unique"):
        Config.from_dict({"models": [{"name": "a", "type": "majority"}, {"name": "a", "type": "nb"}]})


def test_selection_without_validation_split_rejected():
    with pytest.raises(ConfigError, match="validation split"):
        Config.from_dict({"split": {"val_size": 0.0}, "models": [{"name": "a", "type": "majority"}, {"name": "b", "type": "nb"}]})


def test_grouped_requires_group_column():
    with pytest.raises(ConfigError, match="group_column"):
        Config.from_dict({"split": {"strategy": "grouped"}, "models": [{"name": "a", "type": "majority"}]})


def test_hash_is_stable_and_sensitive():
    base = {"models": [{"name": "a", "type": "majority"}]}
    a, b = Config.from_dict(base), Config.from_dict(base)
    assert a.config_hash() == b.config_hash()
    c = Config.from_dict({**base, "train": {"seed": 7}})
    assert c.config_hash() != a.config_hash()


def test_missing_config_file():
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml("does/not/exist.yaml")
