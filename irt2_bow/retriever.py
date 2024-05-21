from abc import ABC

from elasticsearch import Elasticsearch
from irt2.types import VID, MID

from irt2_bow.types import QueryDoc
from irt2_bow.elastic import ES_INDEX


class SimilarMentionsRetriever(ABC):

    def retrieve(
        query_docs: list[QueryDoc],
        n: int,
        ignore_mids: set[MID] | None = None,
    ) -> list[VID]:
        """Based on `query_docs`, this function returns connected closed-world vertices.

        Parameters
        ----------
        `query_docs` : list[QueryDoc]
            The documents that can be used to build a query.

        `n` : int
            Maximum number of results to return.


        `ignore_mids` : set[MID] | None, default: None
            Ignores results that are included in `ignored_mids`.

        Returns
        -------
        list[MID] :
            Matched MIDs. The order is important (highest to lowest).
        """
        pass


class MoreLikeThisMentionsRetriever(SimilarMentionsRetriever):

    es_index: ES_INDEX
    es_client: Elasticsearch

    def __init__(
        self,
        es_index: ES_INDEX,
        es_client: Elasticsearch,
    ):
        super().__init__()
        self.es_index = es_index
        self.es_client = es_client

    def retrieve(self, query_docs: list[str], n: int, ignore_mids: set[MID] | None = None) -> list[VID]:
        ignore_mids = ignore_mids or set()

        # query_data = [qd.data for qd in query_docs]
        body = more_like_this_collapse_search_body(docs=query_docs, ignore_mids=ignore_mids)
        response = self.es_client.search(
            body=body,
            index=self.es_index,
            size=n,
        )

        mids = [hit["_source"]["mid"] for hit in response["hits"]["hits"]]

        # filter duplicates
        mids = list(dict.fromkeys(mids))

        assert len(mids) <= n

        return mids


# ===============
# Queries
# ===============


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


def more_like_this_collapse_search_body_optim(
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
                "filter": {
                    "terms": {
                        "mid": list(ignore_mids),
                    },
                },
            }
        },
        "collapse": {"field": "mid.keyword"},
    }


# ================
# Text Retrievers
# ================
class TextRetriever(ABC):

    def retrieve(
        self,
        query_docs: list[QueryDoc],
        mention: MID,
        n: int,
    ) -> list[str]:
        """Based on `query_docs`, this function returns texts related to a mention.

        Parameters
        ----------
        `query_docs` : list[QueryDoc]
            The documents that can be used to build a query.

        `n` : int
            Maximum number of results to return.

        `mention` : MID
            Only returns texts that belong to the mention.

        Returns
        -------
        list[str] :
            Matched texts.
        """
        pass


class ElasticTextRetriever(TextRetriever):

    index: ES_INDEX
    client: Elasticsearch

    def __init__(self, index: ES_INDEX, client: Elasticsearch):
        super().__init__()

        self.index = index
        self.client = client

    def retrieve(self, query_docs: list[QueryDoc], mention: MID, n: VID) -> list[set]:
        body = order_mention_texts_query(docs=query_docs, mid=mention)
        response = self.client.search(
            body=body,
            index=self.index,
            size=n,
        )

        hits = [hit for hit in response["hits"]["hits"]]

        # sanity check
        for hit in hits:
            assert hit["mid"] == mention

        return [hit["data"] for hit in hits]


def order_mention_texts_query(docs: list[str], mention: MID):
    return {
        "query": {
            "bool": {
                "must": {
                    "more_like_this": {
                        "fields": ["data"],
                        "like": " ".join(docs),
                        "min_term_freq": 1,
                        "max_query_terms": 12,
                    },
                    "match": {"mid": mention},
                },
            }
        }
    }
