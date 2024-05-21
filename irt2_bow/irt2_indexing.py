from typing import Generator
from dataclasses import asdict
from tqdm import tqdm

import click
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from irt2.dataset import IRT2

import irt2_bow
from irt2_bow.elastic import get_client, get_es_index_for_linking, ES_INDEX
from irt2_bow.types import Variant, Split, DatasetName, DatasetConfig
from irt2_bow.utils import dataset_from_config, is_blp, is_irt


def _index_irt2(dataset: IRT2, client: Elasticsearch, index: ES_INDEX):
    print("Indexing closed contexts")

    with dataset.closed_contexts() as ctxs:
        bulk(
            client=client,
            index=index,
            request_timeout=60 * 10,
            actions=_yield_doc(ctxs),
        )


def _index_blp(dataset: IRT2, split: Split, client: Elasticsearch, index: ES_INDEX):

    assert split is not None

    # index closed contexts
    print("Indexing closed contexts")
    with dataset.closed_contexts() as ctxs:
        bulk(
            client=client,
            index=index,
            request_timeout=60 * 10,
            actions=_yield_doc(ctxs),
        )

    if split in {Split.VALID, Split.TEST}:
        # index open val contexts too
        print("Indexing open-valid contexts")
        with dataset.open_contexts_val() as ctxs:
            bulk(
                client=client,
                index=index,
                request_timeout=60 * 10,
                actions=_yield_doc(ctxs),
            )

    if split is Split.TEST:
        # index open test contexts too
        print("Indexing open-test contexts")
        with dataset.open_contexts_test() as ctxs:
            bulk(
                client=client,
                index=index,
                request_timeout=60 * 10,
                actions=_yield_doc(ctxs),
            )


def _yield_doc(contexts: Generator):
    for i, ctx in enumerate(contexts):
        yield asdict(ctx)

        if i % 10000 == 0:
            print(f"indexed {i} docs")


def _get_mappings():
    return {
        "properties": {
            "data": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "mention": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "mid": {"type": "long", "fields": {"keyword": {"type": "keyword"}}},
            "origin": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
        }
    }


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(DatasetName.values()),
    help="Name of the dataset to index",
    required=True,
)
@click.option(
    "--split",
    type=click.Choice(Split.values()),
    help="The variant of dataset to index. Required if dataset is blp",
    required=False,
)
@click.option(
    "--re-create",
    "-f",
    type=bool,
    is_flag=True,
    help="Whether to re-create the index if already exists",
)
def main(dataset_name: str, split: str | None, re_create: bool):

    # input validation
    split: Split | None = Split(split) if dataset_name.startswith("blp") else None
    dataset_name: DatasetName = DatasetName(dataset_name)

    # load dataset
    configs_dir = irt2_bow.ENV.DIR.DATA_CONF
    config_path = configs_dir / f"{Variant.ORIGINAL.value}.yaml"
    config = DatasetConfig.from_yaml(file=config_path, name=dataset_name)
    dataset = dataset_from_config(config=config)

    # connect to elastic
    elastic_client = get_client()
    print("Connected to elasticsearch", elastic_client.info())

    # setup index
    es_index = get_es_index_for_linking(dataset=dataset, split=split)
    print(f"Preparing indexing for index '{es_index}'")

    if elastic_client.indices.exists(index=es_index):
        print(f"Index {es_index} already exists")

        if re_create:
            print("Deleting index ...")
            elastic_client.indices.delete(index=es_index)
        else:
            print("Exiting")
            exit(0)

    elastic_client.indices.create(
        index=es_index,
        mappings=_get_mappings(),
    )

    # run indexing

    print(f"Indexing docs for {dataset.name}")

    if is_irt(dataset):
        _index_irt2(
            dataset=dataset,
            client=elastic_client,
            index=es_index,
        )
    elif is_blp(dataset):
        _index_blp(
            dataset=dataset,
            split=split,
            client=elastic_client,
            index=es_index,
        )

    print(f"Indexing -{dataset.name}- completed")


if __name__ == "__main__":
    main()
