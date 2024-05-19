from typing import Generator
from dataclasses import asdict

import click
from elasticsearch.helpers import bulk

import irt2_bow
from irt2_bow.elastic import get_client, get_es_index_for_linking
from irt2_bow.types import Variant, DatasetName, DatasetConfig
from irt2_bow.utils import dataset_from_config


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
    "--variant",
    type=click.Choice(Variant.values()),
    help="The variant of dataset to index",
    required=True,
)
@click.option(
    "--dataset-name",
    type=click.Choice(DatasetName.values()),
    help="Name of the dataset to index",
    required=True,
)
@click.option(
    "--re-create",
    "-f",
    type=bool,
    is_flag=True,
    help="Whether to re-create the index if already exists",
)
def main(variant: str, dataset_name: str, re_create: bool):

    # input validation
    variant: Variant = Variant(variant)
    dataset_name: DatasetName = DatasetName(dataset_name)

    # load dataset
    configs_dir = irt2_bow.ENV.DIR.DATA_CONF
    config_path = configs_dir / f"{variant.value}.yaml"
    config = DatasetConfig.from_yaml(file=config_path, name=dataset_name)
    dataset = dataset_from_config(config=config)

    # connect to elastic
    elastic_client = get_client()
    print("Connected to elasticsearch", elastic_client.info())

    # setup index
    es_index = get_es_index_for_linking(dataset=dataset, variant=variant)
    print(f"Preparing indexing for index '{es_index}'")

    if elastic_client.indices.exists(index=es_index):
        print(f"Index {es_index} already exists")

        if re_create:
            print("Deleting index ...")
            elastic_client.indices.delete(index=es_index)
        else:
            print("Exiting")
            exit(0)

    elastic_client.indices.create(index=es_index, mappings=_get_mappings())

    # run indexing

    print(f"Indexing docs for {dataset.name}")
    with dataset.closed_contexts() as closed_ctxs:
        bulk(
            client=elastic_client,
            index=es_index,
            request_timeout=60 * 10,
            actions=_yield_doc(closed_ctxs),
        )

    # if variant is full, we also index the validation contexts
    if variant is Variant.FULL:
        print("Indexing the validation contexts")
        with dataset.open_contexts_val() as val_ctxs:
            bulk(
                client=elastic_client,
                index=es_index,
                request_timeout=60 * 10,
                actions=_yield_doc(val_ctxs),
            )

    print(f"Indexing -{dataset.name}- completed")


if __name__ == "__main__":
    main()
