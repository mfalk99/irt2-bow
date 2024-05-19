from abc import ABC
import csv
import random
import yaml
import json

import click
from elasticsearch import Elasticsearch
from tqdm import tqdm

from irt2.dataset import IRT2
from irt2.types import RID, MID, VID

from irt2_bow.types import RetrievedDoc, QueryDoc
from irt2_bow.types import IRT2Size, Split, RankingTask
from irt2_bow.utils import load_irt2
from irt2_bow.elastic import ES_INDEX, more_like_this_collapse_search_body, get_client, ranking_index_for_irt2
from irt2_bow.utils import get_heads, get_tails, extract_query_documents, get_docs_for_mids


# NOTE: Check this file

DEFAULT_MAX_QUERY_DOC = 100


class RetrieveStrategy(ABC):
    def retrieve(query_docs: dict[MID, list[QueryDoc]]) -> dict[MID, RetrievedDoc]:
        pass


class MoreLikeThisRetrieveStrategy(RetrieveStrategy):

    es_index: ES_INDEX
    es_client: Elasticsearch

    def __init__(self, es_index: ES_INDEX, es_client: Elasticsearch):
        self.es_index = es_index
        self.es_client = es_client

    def _get_unique_hits(self, es_response: dict) -> dict[MID, RetrievedDoc]:
        result = dict()

        for hit in es_response["hits"]["hits"]:

            doc = RetrievedDoc.from_es_hit(hit=hit, assert_index=self.es_index)
            assert doc.mid not in result
            result[doc.mid] = doc

        return result

    def retrieve(
        self,
        query_docs: "dict[MID, list[QueryDoc]]",
    ) -> dict[MID, RetrievedDoc]:

        found = dict()

        qdocs = [doc for docs in query_docs.values() for doc in docs]

        while True:

            es_resp = None
            body = more_like_this_collapse_search_body(
                docs=qdocs,
                ignore_mids=set(found),
            )

            es_resp = self.es_client.search(body=body, size=10000, index=self.es_index)

            hits = self._get_unique_hits(es_response=es_resp)

            for hit_mid in hits:
                assert hit_mid not in found
            found.update({mid: d for mid, d in hits.items()})

            if len(hits) < 10000:
                break
        return found


def store_log(log: str, out: str):
    with open(out, "w") as log_fd:
        yaml.safe_dump(log, log_fd)


# Write output to csv
def write_row(
    rid: RID,
    vid: VID,
    csv_writer: csv.writer,
    scores: "dict[MID, float]",
    all_ow_mids: "set[str]",
):
    # [mid1, score1, mid2, score2, ...]
    predictions = [x for mid, score in scores.items() for x in (mid, score)]

    row = [vid, rid, *predictions]
    assert (len(row) - 2) / 2 == len(all_ow_mids)

    csv_writer.writerow(row)


class RankingBaseline:

    split: IRT2Size
    irt2: IRT2

    retriever: RetrieveStrategy
    max_query_docs: int

    def __init__(
        self,
        size: IRT2Size,
        irt2: IRT2,
        retriever: RetrieveStrategy,
        max_query_docs: int,
    ):
        self.size = size
        self.irt2 = irt2

        self.retriever = retriever
        self.max_query_docs = max_query_docs

    def run_ranking(self, split: Split, task: RankingTask, out: str):

        ranking_tasks = self.get_tasks(split=split, task=task)
        ow_mids = self.get_ow_mids(split=split)

        assert ranking_tasks is not None
        assert ow_mids is not None

        print("Gathering documents for building queries ...")
        cw_mids = set.union(*self.irt2.closed_mentions.values())

        cw_docs = self.preselect_query_docs(
            mids=cw_mids,
            max_per_mid=self.max_query_docs,
        )

        assert cw_mids is not None
        assert len(cw_docs) == len(cw_mids)

        out_fd = open(out, "w")
        csv_writer = csv.writer(out_fd, delimiter=",")

        for vid, rid in tqdm(ranking_tasks, desc=f"Running {task} tasks"):

            scores = None
            query_vids = None
            query_mids = set()
            query_docs = None

            if task == RankingTask.HEADS:
                query_vids = get_heads(vid=vid, rid=rid, irt2=self.irt2)
            elif task == RankingTask.TAILS:
                query_vids = get_tails(vid=vid, rid=rid, irt2=self.irt2)
            assert query_vids is not None

            for _vid in query_vids:
                query_mids.update(self.irt2.closed_mentions[_vid])

            query_docs = extract_query_documents(
                docs=cw_docs,
                mids=query_mids,
                max_docs=self.max_query_docs,
            )
            assert query_docs is not None

            scores = self.get_ranking_scores(query_docs=query_docs, all_ow_mids=ow_mids)
            assert scores is not None

            write_row(
                vid=vid,
                rid=rid,
                csv_writer=csv_writer,
                scores=scores,
                all_ow_mids=ow_mids,
            )

        out_fd.close()

    def get_tasks(self, split: Split, task: RankingTask):
        task_choices = {
            Split.VALID: {
                RankingTask.HEADS: self.irt2.open_ranking_val_heads,
                RankingTask.TAILS: self.irt2.open_ranking_val_tails,
            },
            Split.TEST: {
                RankingTask.HEADS: self.irt2.open_ranking_test_heads,
                RankingTask.TAILS: self.irt2.open_ranking_test_tails,
            },
        }

        assert task_choices[split][task] is not None
        return task_choices[split][task]

    def get_ow_mids(self, split: Split):
        mids_choices = {
            Split.VALID: self.irt2.open_mentions_val,
            Split.TEST: self.irt2.open_mentions_test,
        }

        assert mids_choices[split] is not None
        return set.union(*mids_choices[split].values())

    def preselect_query_docs(self, mids: set[MID], max_per_mid: int):
        with self.irt2.closed_contexts() as contexts:
            docs = get_docs_for_mids(
                mids=mids,
                max_docs_per_mid=max_per_mid,
                contexts=contexts,
            )
        return docs

    def get_ranking_scores(
        self,
        query_docs: dict[MID, list[str]],
        all_ow_mids: set[MID],
    ):
        retrieved_docs = self.retriever.retrieve(query_docs)
        scores = {doc.mid: doc.score for doc in retrieved_docs.values()}

        found_mids = set(scores)
        missing_mids = all_ow_mids.difference(found_mids)
        assert len(missing_mids) + len(found_mids) == len(
            all_ow_mids
        ), f"{len(missing_mids)} (missing) + {len(found_mids)} (found) != {len(all_ow_mids)} (all)"

        # Add 0 values for all missing open-world mids.
        # If vid is a true open-world vid, the scores for all mids are 0.
        random.shuffle(list(missing_mids))
        scores.update({mid: 0 for mid in missing_mids})
        assert len(scores) == len(all_ow_mids)

        return scores


@click.command()
@click.option(
    "--size",
    type=click.Choice(IRT2Size.values()),
    required=True,
    help="dataset size to execute",
)
@click.option(
    "--split",
    type=click.Choice(Split.values()),
    required=True,
    help="one of 'validation' or 'test'",
)
@click.option(
    "--task",
    type=str,
    required=click.Choice(RankingTask.values()),
    help="one of 'heads' or 'tails'",
)
@click.option(
    "--max-query-docs",
    type=int,
    required=False,
    help="number of documents used for building a query",
    default=DEFAULT_MAX_QUERY_DOC,
)
@click.option(
    "--seed",
    type=int,
    help="Random seed",
)
@click.option(
    "--out",
    type=click.Path(),
    required=False,
    default=None,
    help="optional output file for csv",
)
def main(
    size: str,
    split: str,
    task: str,
    max_query_docs: int,
    seed: int,
    out: str | None,
):
    # input validation
    assert max_query_docs > 0
    size: IRT2Size = IRT2Size(size)
    split: Split = Split(split)
    task: RankingTask = RankingTask(task)

    # set seed
    print(f"Setting seed to {seed}")
    random.seed(seed)

    # load dataset
    print("Loading dataset ...")
    irt2 = load_irt2(size)

    es_index = ranking_index_for_irt2(irt2)
    es_client = get_client()

    log = {
        "options": {
            "size": size.value,
            "split": split.value,
            "max_query_docs": max_query_docs,
        },
        "irt2": {
            "dataset_name": irt2.name,
            "irt_info": str(irt2),
        },
        "elastic": {"index_used": es_index},
        "out_path": out,
    }
    store_log(log, out=f"{out}.log.yaml")
    print(json.dumps(log, indent=2))

    ranking = RankingBaseline(
        size=size,
        irt2=irt2,
        retriever=MoreLikeThisRetrieveStrategy(es_index=es_index, es_client=es_client),
        max_query_docs=max_query_docs,
    )

    ranking.run_ranking(split=split, task=task, out=out)


if __name__ == "__main__":
    main()
