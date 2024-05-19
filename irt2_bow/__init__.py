from pathlib import Path

_root = Path(__file__).parent.parent


class _DIR:
    DATA_PATH = _root.parent
    DATA_CONF = _root.parent / "irt2" / "conf" / "datasets"
    EXPERIMENT_SEED = 31189


class ENV:
    DIR = _DIR
