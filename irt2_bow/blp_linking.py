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
    load_mid2texts,
)
from irt2_bow.elastic import get_client, es_index_by_mention_splits
from irt2_bow.types import DatasetName
from irt2_bow.types import Split, LinkingTask, MentionSplit
from irt2_bow.retriever import SimilarMentionsRetriever, MoreLikeThisMentionsRetriever

"""blp_linking.py

Bag-of-words (Bow) kgc baseline for the blp datasets.
Due to the blp dataset containing semi-inductive and fully-inductive samples, the pipline proposed for the irt2 datasets performs poorly,
as only transductive samples can be obtained.

Here, the Bow baseline consists of 3 steps:
(1) Given a task mention m_t and relation r_t, the texts associated with m_t are used to query for similar mentions (m_s1, ..., m_sn) in the training set using bm25
(2) The vertices of the similar mentions are used to obtain the known heads/tails vertices v_r1_1, ..., v_rn_p based on r_t
(3) Texts associated with the known head/tail vertices are used to query vertices in the test set

"""


# The maximum number of docs used for building the similar queries
DEFAULT_SIM_QUERY_DOCS = 10

# The maximum number of similar mentions obtained in step (1)
DEFAULT_SIM_MENTIONS_RETRIEVED = 100

# The maximum number of docs used to build the target query
DEFAULT_TARGET_QUERY_DOCS = 10

# The maximum number of target mentions obtained in step (3)
DEFAULT_TARGET_MENTIONS_RETRIEVED = 100


class BLPLinkingBaseline:

    dataset: IRT2

    similar_mention_retriever: SimilarMentionsRetriever
    target_mention_retriever: SimilarMentionsRetriever

    def __init__(
        self,
        dataset: IRT2,
        similar_mention_retriever: SimilarMentionsRetriever,
        target_mention_retriever: SimilarMentionsRetriever,
    ):
        self.dataset = dataset

        self.similar_mention_retriever = similar_mention_retriever
        self.target_mention_retriever = target_mention_retriever

    def run_linking(
        self,
        split: Split,
        task: LinkingTask,
        max_similar_query_docs: int,
        max_similar_mentions: int,
        max_target_query_docs: int,
        max_target_mentions: int,
        mid2texts: dict[MID, list[str]],
    ):
        linking_tasks = self.get_tasks(split=split, task=task)

        # setup
        train_vids = self._select_train_vids(split)
        target_vids = self._select_target_vids(split)
        train_mid2vid = self._build_train_mid2vid_mapping(split)
        train_vid2mids = self._build_train_vid2mids_mapping(split)
        test_mid2vid = self._build_test_mid2vid_mapping(split)

        for mid, rid in tqdm(linking_tasks, desc=f"Running {task} tasks"):

            if mid not in mid2texts:
                # edge case
                # no associated text that belong to the tasks mid
                yield (mid, rid), dict()
                continue

            # (1) Step
            # query for similar mentions / vertices
            task_mention_docs = random.sample(mid2texts[mid], k=min(max_similar_query_docs, len(mid2texts[mid])))
            similar_mentions = self.similar_mention_retriever.retrieve(
                query_docs=task_mention_docs,
                n=max_similar_mentions,
            )

            # map mentions to vertices
            similar_vertices = [train_mid2vid[mid] for mid in similar_mentions]
            # make sure that only vertices from the training set are selected
            similar_vertices = [vid for vid in similar_vertices if vid in train_vids]

            # filter duplicate vids
            similar_vertices = list(dict.fromkeys(similar_vertices))

            # (2) Step
            # select connected nodes
            connected_vertices = self.get_connected_vertices(
                vids=similar_vertices,
                rid=rid,
                task=task,
            )
            connected_vertices = [vid for vids in connected_vertices for vid in vids]
            connected_vertices = list(dict.fromkeys(connected_vertices))

            # (3) Step
            # search for target mentions / vertices using docs from similar mentions
            # obtained from the training set
            target_query_mentions = [train_vid2mids[vid] for vid in connected_vertices]
            target_query_mentions = [mid for mids in target_query_mentions for mid in mids]

            # filter mentions that do not have associated text
            target_query_mentions = [mid for mid in target_query_mentions if mid in mid2texts]

            target_query_docs = [text for mid in target_query_mentions for text in mid2texts[mid]]
            target_query_docs = target_query_docs[:max_target_query_docs]
            # target_query_docs = random.sample(target_query_docs, k=min(max_target_query_docs, len(target_query_docs)))

            scores = dict()
            if len(target_query_docs) == 0:
                # edge case
                # could not find similar connected vertices in the training set
                pass
            else:
                target_mentions = self.target_mention_retriever.retrieve(
                    query_docs=target_query_docs, n=max_target_mentions
                )
                # has to be limited to test_mid2vids for subsampling
                target_vertices = [test_mid2vid[mid] for mid in target_mentions]  # if mid in test_mid2vid]

                # filter vertices that do not belong to the target set (should not change anything)
                target_vertices = [vid for vid in target_vertices if vid in target_vids]
                # filter duplicate vertices
                target_vertices = list(dict.fromkeys(target_vertices))

                # build scores based on the order of the vertices found
                scores = build_scores(found=target_vertices)

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

    def _select_train_vids(self, split: Split) -> set[VID]:
        train_vids = set(self.dataset.closed_mentions)
        validation_vids = set(self.dataset.open_mentions_val)

        if split is Split.VALID:
            return train_vids
        if split is Split.TEST:
            return train_vids | validation_vids

        raise ValueError("unexpected setup")

    def _select_target_vids(self, split) -> set[VID]:
        train_vids = set(self.dataset.closed_mentions)
        valid_vids = set(self.dataset.open_mentions_val)
        test_vids = set(self.dataset.open_mentions_test)

        if split is Split.VALID:
            return train_vids | valid_vids
        if split is Split.TEST:
            return train_vids | valid_vids | test_vids

        raise ValueError("unexpected setup")

    def _build_train_mid2vid_mapping(self, split: Split):
        mid2vid_train = self.dataset.idmap.mid2vid[IRT2Split.train]
        mid2vid_valid = self.dataset.idmap.mid2vid[IRT2Split.valid]

        if split is Split.VALID:
            return mid2vid_train
        elif split is Split.TEST:
            return mid2vid_train | mid2vid_valid

        raise ValueError("unknown setup")

    def _build_train_vid2mids_mapping(self, split: Split):
        vid2mids_train = self.dataset.idmap.vid2mids[IRT2Split.train]
        vid2mids_valid = self.dataset.idmap.vid2mids[IRT2Split.valid]

        if split is Split.VALID:
            return vid2mids_train
        elif split is Split.TEST:
            return vid2mids_train | vid2mids_valid

        raise ValueError("unknown setup")

    def _build_test_mid2vid_mapping(self, split: Split):
        mid2vid_train = self.dataset.idmap.mid2vid[IRT2Split.train]
        mid2vid_valid = self.dataset.idmap.mid2vid[IRT2Split.valid]
        mid2vid_test = self.dataset.idmap.mid2vid[IRT2Split.test]

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


def build_scores(found: list[VID]) -> dict[VID, float]:
    scores = dict()

    # The score of a document depends on the outer index in `found`.
    # A higher position in the outer list leads to a higher score.

    for i, vid in enumerate(found):
        # i = 0 => vid: 1.0, i = 1 => vid: 0.5, ...
        # s = {vid: 1 / (i + 1) for vid in found[i] if vid not in scores}
        score = 1 / (i + 1)

        if vid not in scores:
            scores[vid] = score

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
    "--num-sim-query-docs",
    type=int,
    help="max number of docs to use for the similar mentions queries",
    default=DEFAULT_SIM_QUERY_DOCS,
    required=False,
)
@click.option(
    "--num-sim-mentions-retrieved",
    type=int,
    help="max number of similar mentions queried",
    default=DEFAULT_SIM_MENTIONS_RETRIEVED,
    required=False,
)
@click.option(
    "--num-target-query-docs",
    type=int,
    help="max number of docs to use for the target queries",
    default=DEFAULT_TARGET_QUERY_DOCS,
)
@click.option(
    "--num-target-mentions-retrieved",
    type=int,
    help="max number of target mentions retrieved",
    default=DEFAULT_TARGET_MENTIONS_RETRIEVED,
)
def main(
    dataset_name: str,
    split: str,
    task: str,
    with_subsampling: bool,
    out: str,
    num_sim_query_docs: int,
    num_sim_mentions_retrieved: int,
    num_target_query_docs: int,
    num_target_mentions_retrieved: int,
):
    # input validation
    assert num_sim_query_docs > 0
    assert num_sim_mentions_retrieved > 0
    assert num_target_query_docs > 0
    assert num_target_mentions_retrieved > 0

    dataset_name: DatasetName = DatasetName(dataset_name)
    split: Split = Split(split)
    task: LinkingTask = LinkingTask(task)

    assert dataset_name in {DatasetName.FB, DatasetName.WIKI, DatasetName.WN}

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
    es_client = get_client()

    # setup retriever

    #   sim(ilar) mentions retriever
    #   looks only in the training splits for similar mentions

    sim_index = es_index_by_mention_splits(
        dataset=dataset,
        splits=(
            {
                MentionSplit.TRAIN,
            }
            if split is Split.VALID
            else {
                MentionSplit.TRAIN,
                MentionSplit.VALID,
            }
        ),
    )
    sim_retriever = MoreLikeThisMentionsRetriever(
        es_index=sim_index,
        es_client=es_client,
    )

    #   the target index looks for similar target vertices
    target_index = es_index_by_mention_splits(
        dataset=dataset,
        splits=(
            {MentionSplit.TRAIN, MentionSplit.VALID}
            if split is Split.VALID
            else {
                MentionSplit.TRAIN,
                MentionSplit.VALID,
                MentionSplit.TEST,
            }
        ),
    )
    target_retriever = MoreLikeThisMentionsRetriever(
        es_index=target_index,
        es_client=es_client,
    )

    # log experiment config
    log = {
        "task": task.value,
        "options": {
            "max_sim_query_docs": num_sim_query_docs,
            "max_sim_mentions_returned": num_sim_mentions_retrieved,
            "max_target_query_docs": num_target_query_docs,
            "max_target_mentions_returned": num_target_mentions_retrieved,
        },
        "dataset": {
            "dataset_name": dataset.name,
            "data": str(dataset),
            "with_subsampling": with_subsampling,
        },
        "elastic": {
            "similar_mentions_index": sim_index,
            "target_mentions_index": target_index,
        },
        "out_path": out,
    }
    print(json.dumps(log, indent=2))
    store_log(log, out=f"{out}.log.yaml")

    linking = BLPLinkingBaseline(
        dataset=dataset,
        similar_mention_retriever=sim_retriever,
        target_mention_retriever=target_retriever,
    )

    # load mid2texts mapping
    mid2texts = load_mid2texts(dataset=dataset, split=split)

    # prepare output writer
    out_fd = open(out, "w")
    csv_writer = csv.writer(out_fd, delimiter=",")

    for task, scores in linking.run_linking(
        split=split,
        task=task,
        mid2texts=mid2texts,
        max_similar_query_docs=num_sim_query_docs,
        max_similar_mentions=num_sim_mentions_retrieved,
        max_target_query_docs=num_target_query_docs,
        max_target_mentions=num_target_mentions_retrieved,
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
