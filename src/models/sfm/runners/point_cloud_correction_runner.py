import click

from mvs import run_mvs_position_correction
from tqdm.auto import tqdm


@click.command()
@click.argument("scan_id_start", type=int, required=True)
@click.argument("scan_id_end", type=int, required=True)
@click.option(
    "-p",
    "--cloud-compare-exe-path",
    "cloud_compare_path",
    type=click.Path(dir_okay=False),
    required=True,
)
@click.option(
    "-c",
    "--corrected",
    "corrected",
    type=bool,
    default=False,
)
def main(scan_id_start: int, scan_id_end: int, cloud_compare_path: str, corrected: bool):
    click.echo(
        f"\n==============MVS POSITION CORRECTION from {scan_id_start} to {scan_id_end}==============="
    )
    click.echo(f"CloudCompare.exe path: {cloud_compare_path}")
    click.echo(f"Corrected: {corrected}")
    for scan_id in tqdm(range(scan_id_start, scan_id_end + 1)):
        run_mvs_position_correction(scan_id, cloud_compare_path, corrected)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter