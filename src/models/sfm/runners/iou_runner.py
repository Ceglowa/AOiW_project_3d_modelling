import click

from tqdm.auto import tqdm
from mvs import get_mvs_result_vox_path, get_mvs_truth_vox_path
import numpy as np

from sfm_utils import get_iou, is_correct_scan_id, read_voxel

@click.command()
@click.argument("scan_id_start", type=int, required=True)
@click.argument("scan_id_end", type=int, required=True)
@click.option(
    "-c",
    "--corrected",
    "corrected",
    type=bool,
    default=True,
)
@click.option(
    "-m",
    "--maximized",
    "maximized",
    type=bool,
    default=True,
)
@click.option(
    "-v",
    "--verbose",
    "verbose",
    type=bool,
    default=True,
)
def main(scan_id_start: int, scan_id_end: int, corrected: bool, maximized: bool, verbose: bool) -> np.ndarray:
    click.echo(
        f"\n==============MVS CALCULATE IOU from {scan_id_start} to {scan_id_end}==============="
    )
    click.echo(f"Corrected: {corrected}")
    click.echo(f"Maximized: {maximized}")
    ious = []
    for scan_id in range(scan_id_start, scan_id_end + 1):
        if is_correct_scan_id(scan_id):
            result = read_voxel(get_mvs_result_vox_path(scan_id, corrected, maximized))
            truth = read_voxel(get_mvs_truth_vox_path(scan_id, corrected))
            iou = get_iou(result.data, truth.data)
            ious.append(iou)
            if verbose:
                click.echo(f"IOU {scan_id}: {iou}")
    ious = np.array(ious)
    if verbose:
        click.echo(f"Mean {ious.mean():.2f}")

    return ious

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter