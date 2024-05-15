from typing import Generator
from dataclasses import asdict

import click
from elasticsearch.helpers import bulk

from irt2_bow.elastic import get_client, index_for_irt2
from irt2_bow.types import IRT2Split
from irt2_bow.utils import load_irt2


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
    "--size",
    type=click.Choice({"tiny", "small", "medium", "large"}),
    help="What irt2 size to index",
    required=True,
)
@click.option(
    "--re-create",
    "-f",
    type=bool,
    is_flag=True,
    help="Whether to re-create the index if already exists",
)
def main(size: str, re_create: bool):

    # load dataset
    split = IRT2Split(size)
    dataset = load_irt2(split)

    # connect to elastic
    elastic_client = get_client()
    print("Connected to elasticsearch", elastic_client.info())

    # setup index
    es_index = index_for_irt2(dataset)

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

    print(f"Indexing docs for {split}")
    with dataset.closed_contexts() as ctxs:
        bulk(
            client=elastic_client,
            index=es_index,
            request_timeout=60 * 10,
            actions=_yield_doc(ctxs),
        )

    print(f"Indexing {split} completed")


if __name__ == "__main__":
    main()
