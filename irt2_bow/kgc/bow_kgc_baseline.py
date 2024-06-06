import csv
import json
import random
import yaml

import click
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
from irt2_bow.elastic import get_client, get_es_index_for_linking, es_index_by_mention_splits
from irt2_bow.types import DatasetName
from irt2_bow.types import Split, LinkingTask, MentionSplit
from irt2_bow.retriever import SimilarMentionsRetriever, MoreLikeThisMentionsRetriever

# The maximum number of docs used for building a query
DEFAULT_MAX_QUERY_DOCS = 10

# The number of MIDs retrieved per query
DEFAULT_MAX_RETRIEVED_MIDS = 100


class BowKGCBaseline:

    dataset: IRT2

    retriever: SimilarMentionsRetriever

    def __init__(
        self,
        dataset: IRT2,
        retriever: SimilarMentionsRetriever,
    ):
        self.dataset = dataset

        self.retriever = retriever

    def run_linking(self, split: Split, task: LinkingTask, max_query_docs: int, max_retrieved_mentions: int):

        linking_tasks = self.get_tasks(split=split, task=task)

        # cache docs for building queries
        ow_contexts = self.get_ow_contexts(split=split)
        task_mids = {mid for mid, _ in linking_tasks}
        ow_query_docs = self.extract_docs_from_contexts(
            mids=task_mids,
            max_per_mid=max_query_docs,
            contexts=ow_contexts,
        )

        assert len(ow_query_docs) == len(task_mids)

        # select valid vids that can be selected from
        valid_vids = self._select_vids(split)
        mid2vid = self._build_mid2vid_mapping(split)

        for mid, rid in tqdm(linking_tasks, desc=f"Running {task} tasks"):

            # query for similar mentions / vertices
            query_docs = ow_query_docs[mid]
            similar_mentions = self.retriever.retrieve(
                query_docs=query_docs,
                n=max_retrieved_mentions,
                # ignore_mids={mid},
            )

            # map mentions to vertices
            # make sure that only valid, i.e., known vertices are selected
            similar_vertices = [mid2vid[mid] for mid in similar_mentions]
            similar_vertices = [vid for vid in similar_vertices if vid in valid_vids]

            # filter duplicate vids
            similar_vertices = list(dict.fromkeys(similar_vertices))

            # select connected nodes
            connected_vertices = self.get_connected_vertices(
                vids=similar_vertices,
                rid=rid,
                task=task,
            )

            # build scores based on the order of the vertices found
            scores = build_scores(found=connected_vertices)

            yield (mid, rid), scores

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
        train_vids = set(self.dataset.closed_mentions)
        validation_vids = set(self.dataset.open_mentions_val)
        # test_vids = set(self.dataset.open_mentions_test)

        if is_irt(self.dataset):
            return train_vids

        if is_blp(self.dataset):
            if split is Split.VALID:
                return train_vids  # | validation_vids
            if split is Split.TEST:
                return train_vids | validation_vids  # | test_vids

        raise ValueError("unexpected setup")

    def _build_mid2vid_mapping(self, split: Split):
        mid2vid_train = self.dataset.idmap.mid2vid[IRT2Split.train]
        mid2vid_valid = self.dataset.idmap.mid2vid[IRT2Split.valid]
        mid2vid_test = self.dataset.idmap.mid2vid[IRT2Split.test]

        if is_irt(self.dataset):
            return mid2vid_train
        elif is_blp:
            if split is Split.VALID:
                return mid2vid_train | mid2vid_valid
            elif split is Split.TEST:
                return mid2vid_train | mid2vid_valid | mid2vid_test

        raise ValueError("unknown setup")

    def get_connected_vertices(
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


def build_scores(found: list[set[VID]]) -> dict[VID, float]:
    scores = dict()

    # The score of a document depends on the outer index in `found`.
    # A higher position in the outer list leads to a higher score.

    for i in range(len(found)):
        # i = 0 => vid: 1.0, i = 1 => vid: 0.5, ...
        s = {vid: 1 / (i + 1) for vid in found[i] if vid not in scores}

        scores.update(s)

    return scores


def write_row(
    mid: MID,
    rid: RID,
    csv_writer: csv.writer,
    scores: dict[VID, float],
):

    # [vid1, score1, vid2, score2, ...]
    predictions = [x for vid, score in scores.items() for x in (vid, score)]

    row = [mid, rid, *predictions]

    csv_writer.writerow(row)


def store_log(log: any, out: str):
    with open(out, "w") as log_fd:
        yaml.safe_dump(log, log_fd)


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(DatasetName.values()),
    help="name of the dataset",
    required=True,
)
@click.option(
    "--split",
    type=click.Choice(Split.values()),
    help="what split to run",
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
    help="whether to load the subsampled dataset",
)
@click.option(
    "--out",
    type=str,
    help="path where to store the output csv",
    required=True,
)
@click.option(
    "--num-query-docs",
    type=int,
    help="max number of docs to use for query building",
    default=DEFAULT_MAX_QUERY_DOCS,
    required=False,
)
@click.option(
    "--num-retrieved-mentions",
    type=int,
    help="max number of unique mentions queried",
    default=DEFAULT_MAX_RETRIEVED_MIDS,
    required=False,
)
def main(
    dataset_name: str,
    split: str,
    task: str,
    with_subsampling: bool,
    out: str,
    num_query_docs: int,
    num_retrieved_mentions: int,
):
    # input validation
    assert num_query_docs > 0
    assert num_retrieved_mentions > 0

    dataset_name: DatasetName = DatasetName(dataset_name)
    split: Split = Split(split)
    task: LinkingTask = LinkingTask(task)

    # load config
    print("Loading dataset ...")
    dataset_config = get_dataset_config(name=dataset_name, with_subsampling=with_subsampling)

    # set seed
    seed = dataset_config.seed
    print(f"Setting seed to {seed}")
    random.seed(seed)

    # load dataset
    dataset = dataset_from_config(dataset_config)
    print(f"Loaded {str(dataset)} from {dataset_config.path}")

    # init elastic
    es_index = None
    if is_irt(dataset):
        es_index = es_index_by_mention_splits(
            dataset=dataset,
            splits={MentionSplit.TRAIN},
            subsample=with_subsampling,
        )
    elif is_blp(dataset):
        if split is Split.VALID:
            es_index = es_index_by_mention_splits(
                dataset=dataset,
                splits={
                    MentionSplit.TRAIN,
                    MentionSplit.VALID,
                },
                subsample=with_subsampling,
            )
        elif split is Split.TEST:
            es_index = es_index_by_mention_splits(
                dataset=dataset,
                splits={
                    MentionSplit.TRAIN,
                    MentionSplit.VALID,
                    MentionSplit.TEST,
                },
                subsample=with_subsampling,
            )
    es_client = get_client()

    # log experiment config
    log = {
        "task": task.value,
        "options": {
            "max_query_docs": num_query_docs,
            "retrieved_mids": num_retrieved_mentions,
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
    retriever = MoreLikeThisMentionsRetriever(
        es_index=es_index,
        es_client=es_client,
    )
    linking = BowKGCBaseline(
        dataset=dataset,
        retriever=retriever,
    )

    # prepare output writer
    out_fd = open(out, "w")
    csv_writer = csv.writer(out_fd, delimiter=",")

    for task, scores in linking.run_linking(
        split=split,
        task=task,
        max_query_docs=num_query_docs,
        max_retrieved_mentions=num_retrieved_mentions,
    ):

        mid, rid = task
        write_row(
            mid=mid,
            rid=rid,
            csv_writer=csv_writer,
            scores=scores,
        )

    out_fd.close()
    print("Completed")


if __name__ == "__main__":
    main()
