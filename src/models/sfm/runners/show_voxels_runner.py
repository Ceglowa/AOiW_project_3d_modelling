import click

from tqdm.auto import tqdm
from mvs import get_mvs_result_vox_path, get_mvs_truth_vox_path

from sfm_utils import (
    get_iou,
    get_maximized_result_vox_data,
    is_correct_scan_id,
    read_voxel,
    view_voxel,
)


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
def main(scan_id_start: int, scan_id_end: int, corrected: bool, maximized: bool):
    click.echo(
        f"\n==============MVS SHOWING VOXELS from {scan_id_start} to {scan_id_end}==============="
    )
    click.echo(f"Corrected: {corrected}")
    click.echo(f"Maximized: {maximized}")
    for scan_id in tqdm(range(scan_id_start, scan_id_end + 1)):
        if is_correct_scan_id(scan_id):
            view_voxel(get_mvs_truth_vox_path(scan_id, corrected=corrected))
            view_voxel(
                get_mvs_result_vox_path(
                    scan_id, corrected=corrected, maximized=maximized
                )
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter