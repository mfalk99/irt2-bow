from elasticsearch import Elasticsearch

from irt2_bow.types import Split, MentionSplit
from irt2_bow.utils import is_irt, is_blp
from irt2.dataset import IRT2
from irt2.types import MID

_HOST = "http://localhost:9200"

ES_INDEX = str


def get_client() -> Elasticsearch:
    return Elasticsearch(
        hosts=_HOST,
        verify_certs=False,
        timeout=60 * 10,
    )


def ranking_index_for_irt2(dataset: IRT2) -> ES_INDEX:
    raise NotImplementedError()


def get_es_index_for_linking(dataset: IRT2, split: Split | None = None) -> ES_INDEX:

    index_name = None

    if is_irt(dataset):

        if split is None:
            index_name = dataset.name + "-train"
        else:
            index_name = dataset.name + f"-{split.value}"

    elif is_blp(dataset):
        index_name = dataset.name + "-train"

        if split is Split.VALID:
            index_name += "-valid"
        elif split is Split.TEST:
            index_name += "-valid-test"

    assert index_name is not None

    # fix formatting to valid elastic index format
    index_name = index_name.replace("/", "-").lower()

    return index_name


def es_index_by_mention_splits(dataset: IRT2, splits: set[MentionSplit]) -> ES_INDEX:

    available_splits = [MentionSplit.TRAIN, MentionSplit.VALID, MentionSplit.TEST]

    index_name = dataset.name

    for s in available_splits:
        if s in splits:
            index_name += f"-{s.value}"

    # fix formatting to valid elastic index format
    index_name = index_name.replace("/", "-").lower()

    return index_name


def more_like_this_collapse_search_body(
    docs: tuple[str],
    ignore_mids: set[MID] | None = None,
):
    ignore_mids = ignore_mids or set()

    return {
        "query": {
            "bool": {
                "must": {
                    "more_like_this": {
                        "fields": ["data"],
                        "like": " ".join(docs),
                        "min_term_freq": 1,
                        "max_query_terms": 12,
                    }
                },
                "must_not": {
                    "terms": {
                        "mid": list(ignore_mids),
                    },
                },
            }
        },
        "collapse": {"field": "mid.keyword"},
    }
