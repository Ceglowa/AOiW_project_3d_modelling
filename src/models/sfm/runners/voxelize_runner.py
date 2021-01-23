import click

from tqdm.auto import tqdm
from mvs import get_mvs_result_ply_path, get_mvs_result_vox_path, get_mvs_truth_ply_path, get_mvs_truth_vox_path

from sfm_utils import readAndSavePlyToBinvox

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
        f"\n==============MVS VOXELIZE from {scan_id_start} to {scan_id_end}==============="
    )
    click.echo(f"Corrected: {corrected}")
    for scan_id in tqdm(range(scan_id_start, scan_id_end + 1)):
        readAndSavePlyToBinvox(get_mvs_result_ply_path(scan_id, corrected), get_mvs_result_vox_path(scan_id, corrected))
        readAndSavePlyToBinvox(get_mvs_truth_ply_path(scan_id, corrected), get_mvs_truth_vox_path(scan_id, corrected))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter