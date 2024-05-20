from abc import ABC
import csv
import json
import random
import yaml

import click
from elasticsearch import Elasticsearch
from tqdm import tqdm

from irt2.dataset import IRT2
from irt2.types import MID, VID, RID, Split as IRT2Split

from irt2_bow.utils import (
    get_docs_for_mids,
    get_heads,
    get_tails,
    get_dataset_config,
    dataset_from_config,
    is_blp,
    is_irt,
)
from irt2_bow.elastic import ES_INDEX
from irt2_bow.elastic import (
    get_client,
    get_es_index_for_linking,
    more_like_this_collapse_search_body,
)
from irt2_bow.types import Variant, DatasetName
from irt2_bow.types import QueryDoc, Split, LinkingTask

# The maximum number of docs used for building a query
DEFAULT_MAX_QUERY_DOCS = 10

# The number of MIDs retrieved per query
N_RETRIEVED_MIDS = 100

# The Score assigned to vids not found by the retrieval
MISSING_VID_SCORE = 0


class LinkingRetrieveStrategy(ABC):

    top_n_mids: int

    def __init__(self, top_n_mids: int):
        self.top_n_mids = top_n_mids

    def retrieve(
        query_docs: list[QueryDoc],
        dataset: IRT2,
        split_vids: set[VID],
    ) -> list[VID]:
        """Based on `query_docs`, this function returns connected closed-world vertices.

        Parameters
        ----------
        `query_docs` : list[QueryDoc]
            The documents that can be used to build a query.

        `dataset` : IRT2
            Required to map found mids to vids.

        `split_vids`: set[VID]
            Valid vids of the split. All vids returned by this function have to be in `split_vids`.

        Returns
        -------
        list[VID] :
            Connected closed-world vertices.
        """
        pass


class MoreLikeThisLinkingRetrieverStrategy(LinkingRetrieveStrategy):

    top_n_mids: int

    es_index: ES_INDEX
    es_client: Elasticsearch

    def __init__(
        self,
        top_n_mids: int,
        es_index: ES_INDEX,
        es_client: Elasticsearch,
    ):
        super().__init__(top_n_mids=top_n_mids)
        self.es_index = es_index
        self.es_client = es_client

    def retrieve(
        self,
        query_docs: list[str],
        dataset: IRT2,
        split_vids: set[VID],
    ) -> list[VID]:
        # query_data = [qd.data for qd in query_docs]
        body = more_like_this_collapse_search_body(docs=query_docs)
        response = self.es_client.search(
            body=body,
            index=self.es_index,
            size=self.top_n_mids,
        )

        mids = [hit["_source"]["mid"] for hit in response["hits"]["hits"]]
        mapped_vids = [vid for mid in mids for vid in dataset.idmap.mid2vids[mid] if vid in split_vids]

        return mapped_vids


class LinkingBaseline:

    dataset: IRT2

    retriever: LinkingRetrieveStrategy
    max_query_docs: int

    def __init__(
        self,
        dataset: IRT2,
        retriever: LinkingRetrieveStrategy,
        max_query_docs: int,
    ):
        self.dataset = dataset

        self.retriever = retriever
        self.max_query_docs = max_query_docs

    def run_linking(self, split: Split, task: LinkingTask, out: str):

        linking_tasks = self.get_tasks(split=split, task=task)

        # cache docs for building queries
        ow_contexts = self.get_ow_contexts(split=split)
        task_mids = {mid for mid, _ in linking_tasks}
        ow_query_docs = self.extract_docs_from_contexts(
            mids=task_mids,
            max_per_mid=self.max_query_docs,
            contexts=ow_contexts,
        )

        assert len(ow_query_docs) == len(task_mids)

        # select valid vids that can be selected from
        valid_vids = self._select_vids(split)

        # prepare output writer
        out_fd = open(out, "w")
        csv_writer = csv.writer(out_fd, delimiter=",")

        for mid, rid in tqdm(linking_tasks, desc=f"Running {task} tasks"):

            query_docs = ow_query_docs[mid]
            best_vids = self.retriever.retrieve(
                query_docs=query_docs,
                dataset=self.dataset,
                split_vids=valid_vids,
            )

            connected_mentions = self.get_connected_mentions(
                vids=best_vids,
                rid=rid,
                task=task,
            )

            scores = self.build_scores(
                found=connected_mentions,
                all=set(self.dataset.closed_mentions),
            )

            write_row(
                mid=mid,
                rid=rid,
                csv_writer=csv_writer,
                scores=scores,
                dataset=self.dataset,
            )

        out_fd.close()

    def get_tasks(self, split: Split, task: LinkingTask):
        task_choices = {
            Split.VALID: {
                LinkingTask.HEADS: self.dataset.open_kgc_val_heads,
                LinkingTask.TAILS: self.dataset.open_kgc_val_tails,
            },
            Split.TEST: {
                LinkingTask.HEADS: self.dataset.open_kgc_test_heads,
                LinkingTask.TAILS: self.dataset.open_kgc_test_tails,
            },
        }

        assert task_choices[split][task] is not None
        return task_choices[split][task]

    def get_ow_contexts(self, split: Split):
        context_choices = {
            Split.VALID: self.dataset.open_contexts_val,
            Split.TEST: self.dataset.open_contexts_test,
        }

        assert context_choices[split] is not None
        return context_choices[split]

    def extract_docs_from_contexts(self, mids: set[MID], contexts, max_per_mid: int) -> dict[MID, list[str]]:
        with contexts() as ctxs:
            documents = get_docs_for_mids(
                mids=mids,
                max_docs_per_mid=max_per_mid,
                contexts=ctxs,
            )

        return documents

    def _select_vids(self, split: Split) -> set[VID]:
        train_vids = self.dataset.idmap.split2vids[IRT2Split.train]
        validation_vids = self.dataset.idmap.split2vids[IRT2Split.valid]
        test_vids = self.dataset.idmap.split2vids[IRT2Split.test]

        if is_blp(self.dataset):
            if split is Split.VALID:
                return set.union(train_vids, validation_vids)
            if split is Split.TEST:
                return set.union(train_vids, validation_vids, test_vids)

        if is_irt(self.dataset):
            return train_vids

        raise ValueError("unexpected setup")

    def get_connected_mentions(
        self,
        vids: list[VID],
        rid: RID,
        task: LinkingTask,
    ) -> list[set[VID]]:

        get_hits_fn = None
        if task == LinkingTask.HEADS:
            get_hits_fn = get_heads
        elif task == LinkingTask.TAILS:
            get_hits_fn = get_tails
        assert get_hits_fn is not None

        hits = [get_hits_fn(vid=vid, rid=rid, dataset=self.dataset) for vid in vids]

        return hits

    def build_scores(self, found: list[set[VID]], all: set[VID]) -> dict[VID, float]:
        scores = dict()

        # The score of a document depends on the outer index in `found`.
        # A higher position in the outer list leads to a higher score.

        for i in range(len(found)):
            # i = 0 => vid: 1.0, i = 1 => vid: 0.5, ...
            s = {vid: 1 / (i + 1) for vid in found[i]}

            for vid, score in s.items():
                if vid in scores:
                    assert score < scores[vid]

            scores.update(s)

        # Append scores for VIDs not found
        found_vids = {vid for vid in scores if vid in all}
        missing_vids = all.difference(found_vids)

        assert len(found_vids) + len(missing_vids) == len(all)
        assert found_vids | missing_vids == all

        missing_vids = sorted(missing_vids)
        random.shuffle(missing_vids)

        scores.update({vid: MISSING_VID_SCORE for vid in missing_vids})

        return scores


def write_row(
    mid: MID,
    rid: RID,
    csv_writer: csv.writer,
    scores: dict[VID, float],
    dataset: IRT2,
):

    # [vid1, score1, vid2, score2, ...]
    predictions = [x for vid, score in scores.items() for x in (vid, score)]

    row = [mid, rid, *predictions]
    assert (len(row) - 2) / 2 == len(dataset.closed_mentions)

    csv_writer.writerow(row)


def store_log(log: any, out: str):
    with open(out, "w") as log_fd:
        yaml.safe_dump(log, log_fd)


@click.command()
@click.option(
    "--variant",
    type=click.Choice(Variant.values()),
    help="The variant of dataset",
    required=True,
)
@click.option(
    "--dataset-name",
    type=click.Choice(DatasetName.values()),
    help="Name of the dataset",
    required=True,
)
@click.option(
    "--split",
    type=click.Choice(Split.values()),
    help="What split to run",
    required=True,
)
@click.option(
    "--task",
    type=str,
    required=click.Choice(LinkingTask.values()),
    help="one of 'heads' or 'tails'",
)
@click.option(
    "--with-subsampling",
    type=bool,
    is_flag=True,
    help="Whether to load the subsampled dataset",
)
@click.option(
    "--max-query-docs",
    type=int,
    required=False,
    help="Number of docs to use for query building",
    default=DEFAULT_MAX_QUERY_DOCS,
)
@click.option(
    "--out",
    type=str,
    required=True,
    help="path where to store the output csv",
)
def main(
    variant: str,
    dataset_name: str,
    split: str,
    task: str,
    with_subsampling: bool,
    max_query_docs: int,
    out: str | None,
):
    # input validation
    assert max_query_docs > 0
    variant: Variant = Variant(variant)
    dataset_name: DatasetName = DatasetName(dataset_name)
    split: Split = Split(split)
    task: LinkingTask = LinkingTask(task)

    # load config
    print("Loading dataset ...")
    dataset_config = get_dataset_config(name=dataset_name, variant=variant, with_subsampling=with_subsampling)

    # set seed
    seed = dataset_config.seed
    print(f"Setting seed to {seed}")
    random.seed(seed)

    # load dataset
    dataset = dataset_from_config(dataset_config)
    print(f"Loaded {str(dataset)} from {dataset_config.path}")

    # init elastic
    es_index = get_es_index_for_linking(
        dataset=dataset,
        variant=variant,
        split=split,
    )
    es_client = get_client()

    # log experiment config
    log = {
        "task": task.value,
        "options": {
            "max_query_docs": max_query_docs,
            "retrieved_mids": N_RETRIEVED_MIDS,
        },
        "dataset": {
            "dataset_name": dataset.name,
            "data": str(dataset),
            "with_subsampling": with_subsampling,
        },
        "elastic": {"index": es_index},
        "out_path": out,
    }
    print(json.dumps(log, indent=2))
    store_log(log, out=f"{out}.log.yaml")

    # linking
    retriever = MoreLikeThisLinkingRetrieverStrategy(
        top_n_mids=N_RETRIEVED_MIDS,
        es_index=es_index,
        es_client=es_client,
    )
    linking = LinkingBaseline(
        dataset=dataset,
        retriever=retriever,
        max_query_docs=max_query_docs,
    )
    linking.run_linking(split=split, task=task, out=out)


if __name__ == "__main__":
    main()
