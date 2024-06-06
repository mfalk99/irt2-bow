import click
import csv

import irt2.evaluation as eval
from irt2.dataset import IRT2

from irt2_bow.types import Split, DatasetName
from irt2_bow.utils import get_dataset_config, dataset_from_config

DEFAULT_MAX_RANK = 10


def _load_preds(csv_file: str) -> list[tuple[int, int]]:
    rows = csv.reader(open(csv_file))
    return [(int(r[0]), int(r[1])) for r in rows]


def _validate_head_tasks(dataset: IRT2, split: Split, preds_file: str):
    GT_TASK_OPTIONS = {
        Split.VALID: dataset.open_kgc_val_heads,
        Split.TEST: dataset.open_kgc_test_heads,
    }

    gt_tasks = set(GT_TASK_OPTIONS[split])
    pred_tasks = set(_load_preds(preds_file))

    assert gt_tasks == pred_tasks, f"GT:\n{gt_tasks}\n\nPREDS:\n{pred_tasks}"


def _validate_tail_tasks(dataset: IRT2, split: Split, preds_file: str):
    GT_TASK_OPTIONS = {
        Split.VALID: dataset.open_kgc_val_tails,
        Split.TEST: dataset.open_kgc_test_tails,
    }

    gt_tasks = set(GT_TASK_OPTIONS[split])
    pred_tasks = set(_load_preds(preds_file))

    assert gt_tasks == pred_tasks, f"GT:\n{gt_tasks}\n\nPREDS:\n{pred_tasks}"


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

    # validate that there are predictions for every task
    _validate_head_tasks(dataset=dataset, split=split, preds_file=head_task)
    _validate_tail_tasks(dataset=dataset, split=split, preds_file=tail_task)

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
