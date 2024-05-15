from pathlib import Path

_root = Path(__file__).parent.parent


class _DIR:
    IRT2_DATA = _root.parent / "data" / "irt2"


class ENV:
    DIR = _DIR
