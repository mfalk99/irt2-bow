from elasticsearch import Elasticsearch

from irt2_bow.types import Variant, Split
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


def get_es_index_for_linking(dataset: IRT2, variant: Variant, split: Split | None = None) -> ES_INDEX:

    if split is not None and split is Split.VALID:
        # validation splits always run on the original (train-only) texts
        index_name = dataset.name + "-" + Variant.ORIGINAL.value
    else:
        index_name = dataset.name + "-" + variant.name

    # fix formatting to valid elastic index format
    index_name = index_name.replace("/", "-").lower()

    return index_name


# def linking_index_for_irt2(dataset: IRT2, kind: Kind, split: Split) -> ES_INDEX:
# return dataset.name.replace("/", "-").lower()
# index =


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
