import enum
from pathlib import Path
import yaml

from pydantic.dataclasses import dataclass

from irt2.types import MID


# ======================
# Task types
# ======================


class Variant(enum.Enum):
    # has to match the names of the config files in the irt2/conf/datasets dir, except for the subsampling part
    ORIGINAL = "original"
    FULL = "full"

    def values() -> set:
        return {item.value for item in Variant}


class Split(enum.Enum):
    VALID = "validation"
    TEST = "test"

    def values() -> set:
        return {item.value for item in Split}


class MentionSplit(enum.Enum):
    TRAIN = "training"
    VALID = "validation"
    TEST = "testing"

    @staticmethod
    def values() -> set[str]:
        return {item.value for item in MentionSplit}


class RankingTask(enum.Enum):
    HEADS = "heads"
    TAILS = "tails"

    def values() -> set:
        return {item.value for item in RankingTask}


class LinkingTask(enum.Enum):
    HEADS = "heads"
    TAILS = "tails"

    def values() -> set:
        return {item.value for item in LinkingTask}


# ======================
# IRT2 Types
# ======================


class IRT2Size(enum.Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    def values() -> set:
        return {item.value for item in IRT2Size}


# ======================
# Data Loading Config Types
# ======================


class DatasetName(enum.Enum):
    # Values have to match the keys in the yaml config
    FB = "blp/fb15k237"
    WIKI = "blp/wikidata5m"
    WN = "blp/wn18rr"

    IRT2_LARGE = "irt2/large"
    IRT2_MEDIUM = "irt2/medium"
    IRT2_SMALL = "irt2/small"
    IRT2_TINY = "irt2/tiny"

    def values() -> set:
        return {item.value for item in DatasetName}


@dataclass(frozen=True)
class SubsampleConfig:
    test: float
    validation: float
    seed: int | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: str
    path: str
    seed: int
    subsample: SubsampleConfig | None = None
    kwargs: dict | None = None

    @classmethod
    def from_yaml(cls, file: Path | str, name: DatasetName) -> "DatasetConfig":
        config = yaml.safe_load(open(str(file)))

        assert name.value in config["datasets"], f"Dataset {name.value} not in {config['datasets']}"

        seed = config["seed"]
        args = config["datasets"][name.value]

        return cls(seed=seed, name=name, **args)


# ======================
# Elastic Retrieval Types
# ======================


@dataclass(frozen=True)
class QueryDoc:
    mid: MID
    data: str


@dataclass(frozen=True)
class RetrievedDoc:
    mid: MID
    data: str
    score: float

    @staticmethod
    def from_es_hit(hit: dict, assert_index: str):

        mid = hit["_source"]["mid"]
        data = hit["_source"]["data"]
        score = hit["_score"]

        assert mid is not None
        assert data is not None
        assert score is not None

        assert hit["_index"] == assert_index

        return RetrievedDoc(
            mid=mid,
            data=data,
            score=score,
        )
