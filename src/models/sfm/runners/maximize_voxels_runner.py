import click

from tqdm.auto import tqdm
from mvs import get_mvs_result_vox_path, get_mvs_truth_vox_path

from sfm_utils import get_maximized_result_vox_data, read_voxel

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
def main(scan_id_start: int, scan_id_end: int, corrected: bool):
    click.echo(
        f"\n==============MVS MAXIMIZE VOXELIZE from {scan_id_start} to {scan_id_end}==============="
    )
    click.echo(f"Corrected: {corrected}")
    for scan_id in tqdm(range(scan_id_start, scan_id_end + 1)):
        truth = read_voxel(get_mvs_truth_vox_path(scan_id, corrected=corrected))
        result = read_voxel(get_mvs_result_vox_path(scan_id, corrected=corrected))

        _, maximized_result_data, _ = get_maximized_result_vox_data(result.data, truth.data)
        result_maximized = result.clone()
        result_maximized.data = maximized_result_data

        with open(get_mvs_result_vox_path(scan_id, corrected, maximized=True), "wb") as f:
            result_maximized.write(f)



if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter