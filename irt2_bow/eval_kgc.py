import click

import irt2.evaluation as eval

from irt2_bow.types import Split, DatasetName
from irt2_bow.utils import get_dataset_config, dataset_from_config

DEFAULT_MAX_RANK = 10


@click.command()
@click.option(
    "--dataset-name",
    type=click.Choice(DatasetName.values()),
    help="name of the dataset",
    required=True,
)
@click.option(
    "--head-task",
    type=click.Path(exists=True),
    help="path to head predictions (csv)",
    required=True,
)
@click.option(
    "--tail-task",
    type=click.Path(exists=True),
    help="path to tail predictions (csv)",
    required=True,
)
@click.option(
    "--split",
    type=click.Choice(Split.values()),
    help="one of validation or test",
    required=True,
)
@click.option(
    "--with-subsampling",
    type=bool,
    is_flag=True,
    help="whether to load the subsampled dataset",
)
@click.option(
    "--max-rank",
    type=int,
    help="only consider the first n ranks (target filtered)",
    default=DEFAULT_MAX_RANK,
)
@click.option(
    "--model",
    type=str,
    help="name of the model",
    default="unknown",
    required=False,
)
@click.option(
    "--out",
    type=click.Path(),
    help="path where to store the output report",
    required=True,
)
def main(
    dataset_name: str,
    head_task: str,
    tail_task: str,
    split: str,
    with_subsampling: bool,
    max_rank: int,
    model: str,
    out: str,
):
    # input validation
    assert max_rank > 0

    dataset_name: DatasetName = DatasetName(dataset_name)
    split: Split = Split(split)

    # load dataset
    dataset_config = get_dataset_config(name=dataset_name, with_subsampling=with_subsampling)
    dataset = dataset_from_config(dataset_config)
    print(f"Loaded {str(dataset)} from {dataset_config.path}")

    metrics = eval.evaluate(
        dataset,
        task="kgc",
        split=split.value,
        head_predictions=eval.load_csv(head_task),
        tail_predictions=eval.load_csv(tail_task),
        max_rank=max_rank,
    )

    eval.create_report(
        metrics,
        dataset,
        "kgc",
        split.value,
        model=model,
        filenames=dict(
            head_task=head_task,
            tail_task=tail_task,
        ),
        out=out,
    )


if __name__ == "__main__":
    main()
