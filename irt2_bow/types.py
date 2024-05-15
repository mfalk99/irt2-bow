import enum


class IRT2Split(enum.Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    def values() -> set:
        return {item.value for item in IRT2Split}
