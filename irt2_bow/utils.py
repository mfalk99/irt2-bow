from irt2.dataset import IRT2
from irt2.types import VID, RID, MID

from irt2.loader import LOADER
from irt2.dataset import IRT2
import irt2_bow
from irt2_bow.types import DatasetConfig, DatasetName, Variant


def get_dataset_config(name: DatasetName, variant: Variant, with_subsampling: bool) -> DatasetConfig:
    configs_path = irt2_bow.ENV.DIR.DATA_CONF

    config_file = variant.value
    if with_subsampling:
        config_file += "-subsampled"
    config_file += ".yaml"
    config_path = configs_path / config_file

    return DatasetConfig.from_yaml(file=config_path, name=name)


def dataset_from_config(config: DatasetConfig) -> IRT2:
    """Load a dataset from a config.
    function is adapted from irt2.loader.__init__ : from_config
    """

    dataset_file = f"{config.path}-linking" if config.name.startswith("irt2") else config.path
    dataset_path = irt2_bow.ENV.DIR.DATA_PATH / dataset_file

    loader = LOADER[config.loader]
    dataset: IRT2 = loader(
        dataset_path,
        **config.kwargs or dict(),
    )

    dataset.meta["loader"] = config

    # --- filter gt

    if config.subsample:
        sub_options = config.subsample

        if sub_options.seed is not None:
            seed = sub_options.seed
        else:
            seed = config.seed

        dataset = dataset.tasks_subsample_kgc(
            percentage_val=sub_options.validation,
            percentage_test=sub_options.test,
            seed=seed,
        )

    # --- return

    print(f"loaded {str(dataset)}")
    return dataset


def is_irt(dataset: IRT2) -> bool:
    loader_name = dataset.meta["loader"].name
    return loader_name.lower().startswith("irt2")


def is_blp(dataset: IRT2) -> bool:
    loader_name = dataset.meta["loader"].name
    return loader_name.lower().startswith("blp")


def get_heads(vid: VID, rid: RID, dataset: IRT2) -> set[VID]:
    heads = {vid for (vid, _, _) in dataset.graph.select(tails={vid}, edges={rid})}
    return heads


def get_tails(vid: VID, rid: RID, dataset: IRT2) -> set[VID]:
    tails = {vid for (_, vid, _) in dataset.graph.select(heads={vid}, edges={rid})}
    return tails


def get_docs_for_mids(
    mids: set[MID],
    max_docs_per_mid: int,
    contexts,
) -> dict[MID, list[str]]:
    """Extracts up to max_docs_per_mid documents from contexts for each MID in mids.
    The returned document count can be less than max_docs_per_mid.

    Parameters
    ----------
    `mids` : set[MID]
        The MIDs to extract documents for.

    `max_docs_per_mid` : int
        The maximum document count for each MID.

    `contexts` : Generator
        The IRT2 contexts to extract the documents from.
    """

    documents = {mid: [] for mid in mids}

    for context in contexts:

        if context.mid not in mids:
            continue

        if len(documents[context.mid]) >= max_docs_per_mid:
            continue

        documents[context.mid].append(context.data)

    for docs in documents.values():
        assert len(docs) <= max_docs_per_mid

    return documents


def extract_query_documents(
    docs: dict[MID, list[str]],
    mids: set[MID],
    max_docs: int,
) -> dict[MID, list[str]]:
    """Picks up to max_docs documents in total from docs.

    Parameters
    ----------
    `docs` : dict[MID, list[str]]
        The documents to extract from.

    `max_docs` : int
        The maximum count of documents present in the output.
        The real document count can be less than max_docs.
    """

    documents = dict()
    added = 0

    for i in range(max_docs):

        for mid in mids:

            if added == max_docs:
                break

            if mid not in documents:
                documents[mid] = []

            if i >= len(docs[mid]):
                continue

            doc = docs[mid][i]
            documents[mid].append(doc)
            added += 1

    assert added <= max_docs

    return documents
