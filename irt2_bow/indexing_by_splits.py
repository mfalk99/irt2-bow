from typing import Generator
from dataclasses import asdict

import click
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from irt2.dataset import IRT2

import irt2_bow
from irt2_bow.elastic import get_client, es_index_by_mention_splits, ES_INDEX
from irt2_bow.types import Variant, MentionSplit, DatasetName, DatasetConfig
from irt2_bow.utils import dataset_from_config


def _index(dataset: IRT2, splits: set[MentionSplit], client: Elasticsearch, index: ES_INDEX):

    split2ctxs = {
        MentionSplit.TRAIN: dataset.closed_contexts,
        MentionSplit.VALID: dataset.open_contexts_val,
        MentionSplit.TEST: dataset.open_contexts_test,
    }

    for split in splits:

        print(f"Indexing {split.name} contexts")
        with split2ctxs[split]() as ctxs:
            bulk(
                client=client,
                index=index,
                request_timeout=60 * 10,
                actions=_yield_doc(ctxs, split=split.value),
            )


def _yield_doc(contexts: Generator, split: str):
    for i, ctx in enumerate(contexts):
        doc = asdict(ctx)
        yield {
            "split": split,
            **doc,
        }

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
            "split": {
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
    "--splits",
    type=str,
    help="The variant of dataset to index. Seperate mutliple splits by a comma, e.g., train,test",
    required=False,
)
@click.option(
    "--re-create",
    "-f",
    type=bool,
    is_flag=True,
    help="Whether to re-create the index if already exists",
)
def main(dataset_name: str, splits: str | None, re_create: bool):

    # input validation
    splits: set[MentionSplit] = {MentionSplit(split) for split in splits.split(",")}
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
    es_index = es_index_by_mention_splits(dataset=dataset, splits=splits)
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

    print(f"Indexing docs for {dataset.name}, {' '.join([s.value for s in splits])}")
    _index(dataset=dataset, splits=splits, client=elastic_client, index=es_index)
    print(f"Indexing -{dataset.name}- completed")


if __name__ == "__main__":
    main()
