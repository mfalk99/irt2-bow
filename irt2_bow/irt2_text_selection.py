import random
import pickle
import logging
import yaml

import click
from tqdm import tqdm
from irt2.dataset import IRT2
from irt2.types import MID, RID, VID, Split as IRT2Split

from irt2_bow.types import Split, LinkingTask, DatasetName
from irt2_bow.utils import (
    get_dataset_config,
    dataset_from_config,
    get_docs_for_mids,
    get_heads,
    get_tails,
    load_mid2texts,
)
from irt2_bow.elastic import get_client, get_es_index_for_linking
from irt2_bow.retriever import SimilarMentionsRetriever, MoreLikeThisMentionsRetriever
from irt2_bow.retriever import TextRetriever, ElasticTextRetriever

# similar mentions
DEFAULT_SIM_QUERY_DOCS = 10
DEFAULT_SIM_MENTIONS = 100

# ordering
DEFAULT_ORDER_QUERY_DOCS = 10
DEFAULT_ORDER_SELECTED_TEXTS = 30


class TextSelector:

    dataset: IRT2
    sim_retriever: SimilarMentionsRetriever
    text_retriever: TextRetriever

    max_similar_query_docs: int
    max_similar_mentions: int
    max_order_query_docs: int
    max_order_texts: int

    def __init__(
        self,
        dataset: IRT2,
        sim_retriever: SimilarMentionsRetriever,
        text_retriever: TextRetriever,
        max_similar_query_docs: int,
        max_similar_mentions: int,
        max_order_query_docs: int,
        max_order_texts: int,
    ):
        self.dataset = dataset
        self.sim_retriever = sim_retriever
        self.text_retriever = text_retriever

        self.max_similar_query_docs = max_similar_query_docs
        self.max_similar_mentions = max_similar_mentions
        self.max_order_query_docs = max_order_query_docs
        self.max_order_texts = max_order_texts

    def run_text_selection(
        self, split: Split, task: LinkingTask, mid2texts: dict[MID, list[str]]
    ) -> tuple[dict[tuple[MID, RID], list[str]], dict]:

        linking_tasks = self.get_tasks(split=split, task=task)
        valid_vids = list(self.dataset.closed_mentions)

        # prepare output writer
        output = dict[tuple[MID, RID], list[str]]()

        num_fills = 0
        num_selected = 0

        for mid, rid in tqdm(linking_tasks, desc=f"Running {task} tasks"):

            if mid not in mid2texts:
                # edge case: there are no texts for this mention that could be ordered
                # thus, we simply use an empty list
                output[(mid, rid)] = []
                continue

            # query related mentions
            mention_docs = random.sample(mid2texts[mid], k=min(self.max_similar_query_docs, len(mid2texts[mid])))
            similar_mentions = self.sim_retriever.retrieve(
                query_docs=mention_docs,
                n=self.max_similar_mentions,  # , ignore_mids={mid} # there should be no texts in the index that belong to `mid`
            )

            # map related mentions to their vertices
            # make sure to only select known vertices (valid_vids)
            similar_vertices = [self.dataset.idmap.mid2vid[IRT2Split.train][mid] for mid in similar_mentions]
            similar_vertices = [vid for vid in similar_vertices if vid in valid_vids]

            # filter duplicates
            similar_vertices = list(dict.fromkeys(similar_vertices))

            # get known heads / tails
            connected_vertices = self.get_connected_vertices(vids=similar_vertices, rid=rid, task=task)

            connected_mentions = [
                self.dataset.idmap.vid2mids[IRT2Split.train][vid] for vids in connected_vertices for vid in vids
            ]
            connected_mentions = [mid for mids in connected_mentions for mid in mids]  # flatten

            # filter duplicates
            connected_mentions = list(dict.fromkeys(connected_mentions))

            # edge case: mid=0 bug
            connected_mentions = [mid for mid in connected_mentions if mid != 0]

            # select docs from the connected mentions
            # and use the docs to order the tasks' mention docs
            texts = []

            if len(connected_mentions) > 0:
                connected_mention_docs = [doc for mid in connected_mentions for doc in mid2texts[mid]]
                random.shuffle(connected_mention_docs)
                connected_mention_docs = connected_mention_docs[: self.max_order_query_docs]

                texts = self.text_retriever.retrieve(
                    query_docs=connected_mention_docs, mention=mid, n=self.max_order_texts
                )
            else:
                # edge case: no conntected mention was found to build the order query
                # we proceed to randomly fill the
                logging.info("no connected mention for ")
                pass

            texts, fills = _fill_texts(selected=texts, available_docs=mid2texts[mid], up_to=self.max_order_texts)

            num_selected += len(texts)
            num_fills += fills

            assert (mid, rid) not in output
            output[(mid, rid)] = texts

        # sanity check: every task should be represented in the ouput
        for task in linking_tasks:
            assert task in output

        stats = {
            "num_fills": num_fills,
            "num_selected": num_selected,
        }

        return output, stats

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


def _fill_texts(selected: list[str], available_docs: list[str], up_to: int) -> tuple[list[str], int]:

    # filter duplicates
    selected = list(dict.fromkeys(selected))
    candidates = set(available_docs) - set(selected)

    candidates = list(candidates)
    random.shuffle(candidates)

    selected_candidates = candidates[: up_to - len(selected)]

    num_filled = len(selected_candidates)

    return selected + selected_candidates, num_filled


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(DatasetName.values()),
    required=True,
)
@click.option(
    "--split",
    type=click.Choice(Split.values()),
    required=True,
)
@click.option(
    "--task",
    type=click.Choice(LinkingTask.values()),
    required=True,
)
@click.option(
    "--with-subsampling",
    type=bool,
    is_flag=True,
    help="Whether to run the subsampled split",
)
@click.option(
    "--out",
    type=click.Path(),
    help="path to output file",
    required=True,
)
@click.option(
    "--num-sim-query-docs",
    type=int,
    help="max number of docs used to build the similar-mentions query",
    default=DEFAULT_SIM_QUERY_DOCS,
    required=False,
)
@click.option(
    "--num-sim-mentions",
    type=int,
    help="max number of similar mentions returned",
    default=DEFAULT_SIM_MENTIONS,
    required=False,
)
@click.option(
    "--num-order-query-docs",
    type=int,
    help="max number of docs used to build the order query",
    default=DEFAULT_ORDER_QUERY_DOCS,
    required=False,
)
@click.option(
    "--num-order-texts",
    type=int,
    help="max number of ordered docs retrieved",
    default=DEFAULT_ORDER_SELECTED_TEXTS,
    required=False,
)
def main(
    dataset_name: str,
    split: str,
    task: str,
    with_subsampling: bool,
    out: str,
    num_sim_query_docs: int,
    num_sim_mentions: int,
    num_order_query_docs: int,
    num_order_texts: int,
):

    # input validation
    assert num_sim_query_docs > 0
    assert num_sim_mentions > 0
    assert num_order_query_docs > 0
    assert num_order_texts > 0

    dataset_name: DatasetName = DatasetName(dataset_name)
    split: Split = Split(split)
    task: LinkingTask = LinkingTask(task)

    # load config
    print("Loading dataset ...")
    dataset_config = get_dataset_config(
        name=dataset_name,
        with_subsampling=with_subsampling,
    )

    # set seed
    seed = dataset_config.seed
    print(f"Setting seed to {seed}")
    random.seed(seed)

    # load dataset
    dataset = dataset_from_config(dataset_config)

    # load mid2texts mapping
    mid2texts = load_mid2texts(dataset=dataset, split=split)

    # init elastic
    es_client = get_client()

    # select texts
    similar_mentions_index = get_es_index_for_linking(
        dataset=dataset,
    )
    similar_mentions_retriever: SimilarMentionsRetriever = MoreLikeThisMentionsRetriever(
        es_client=es_client,
        es_index=similar_mentions_index,
    )
    print(f"Using index '{similar_mentions_index}' for similar-mentions-selection")

    order_text_index = get_es_index_for_linking(
        dataset=dataset,
        split=split,
    )
    order_texts_retriever: TextRetriever = ElasticTextRetriever(
        index=order_text_index,
        client=es_client,
    )
    print(f"Using index '{order_text_index}' for text-ordering")

    selector = TextSelector(
        dataset=dataset,
        sim_retriever=similar_mentions_retriever,
        text_retriever=order_texts_retriever,
        max_similar_query_docs=num_sim_query_docs,
        max_similar_mentions=num_sim_mentions,
        max_order_query_docs=num_order_query_docs,
        max_order_texts=num_order_texts,
    )

    output, stats = selector.run_text_selection(split=split, task=task, mid2texts=mid2texts)

    print(f"Saving output to '{out}'")
    pickle.dump(output, open(out, "wb"))

    # save details
    details = {
        "task": task.value,
        "options": {
            "num_sim_query_docs": num_sim_query_docs,
            "num_sim_mentions": num_sim_mentions,
            "num_order_query_docs": num_order_query_docs,
            "num_order_texts": num_order_texts,
        },
        "dataset": {
            "dataset_name": dataset.name,
            "data": str(dataset),
            "with_subsampling": with_subsampling,
        },
        "elastic": {
            "similar_mentions_index": similar_mentions_index,
            "order_text_index": order_text_index,
        },
        "out_path": out,
        "stats": {**stats},
    }

    details_out = out + ".yaml"
    print(f"Saving details to {details_out}")
    yaml.safe_dump(details, open(details_out, "w"))


if __name__ == "__main__":
    main()
