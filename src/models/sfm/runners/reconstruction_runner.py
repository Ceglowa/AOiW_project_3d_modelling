import click

from mvs import run_mvs_reconstruction
from tqdm.auto import tqdm

from sfm_utils import is_correct_scan_id

@click.command()
@click.argument("scan_id_start", type=int, required=True)
@click.argument("scan_id_end", type=int, required=True)
def main(scan_id_start: int, scan_id_end: int):
    click.echo(f"\n==============MVS RECONSTRUCTION from {scan_id_start} to {scan_id_end}===============")
    for scan_id in tqdm(range(scan_id_start, scan_id_end + 1)):
        if is_correct_scan_id(scan_id):
            run_mvs_reconstruction(scan_id)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter