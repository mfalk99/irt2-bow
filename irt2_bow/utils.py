from irt2.dataset import IRT2

import irt2_bow
from irt2_bow.types import IRT2Split


def load_irt2(split: IRT2Split):
    dir = irt2_bow.ENV.DIR.IRT2_DATA
    name = "irt2-cde-{split}-linking".format(split=split.value)
    path = dir / name

    print(f"Loading IRT2 - {name} from {path}")
    return IRT2.from_dir(path)
