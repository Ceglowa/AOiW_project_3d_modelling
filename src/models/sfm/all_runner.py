import click
from runners.reconstruction_runner import main as run_reconstruction
from runners.point_cloud_correction_runner import main as run_correction
from runners.voxelize_runner import main as run_voxelization
from runners.maximize_voxels_runner import main as run_maximization
from runners.iou_runner import main as run_iou
import pandas as pd
import os
from settings import REPORTS_DIR
import plotly.express as px


@click.command()
@click.argument("scan_id_start", type=int, required=True)
@click.argument("scan_id_end", type=int, required=True)
@click.option(
    "-r",
    "--reconstruction",
    "reconstruction",
    type=bool,
    default=True,
)
@click.option(
    "-c",
    "--correction",
    "correction",
    type=bool,
    default=True,
)
@click.option(
    "-p",
    "--cloud-compare-exe-path",
    "cloud_compare_path",
    type=click.Path(dir_okay=False),
)
def main(scan_id_start: int, scan_id_end: int, reconstruction: bool, correction: bool, cloud_compare_path: str):
    if reconstruction:
        run_reconstruction.callback(scan_id_start, scan_id_end, False)
    if correction:
        run_correction.callback(scan_id_start, scan_id_end, cloud_compare_path, False)
    
    run_voxelization.callback(scan_id_start, scan_id_end, False)
    run_voxelization.callback(scan_id_start, scan_id_end, True)

    run_maximization.callback(scan_id_start, scan_id_end, False)
    run_maximization.callback(scan_id_start, scan_id_end, True)

    iou_cf_mf = run_iou.callback(scan_id_start, scan_id_end, False, False, False)
    iou_cf_mt = run_iou.callback(scan_id_start, scan_id_end, False, True, False)
    iou_ct_mf = run_iou.callback(scan_id_start, scan_id_end, True, False, False)
    iou_ct_mt = run_iou.callback(scan_id_start, scan_id_end, True, True, False)

    df = pd.DataFrame({"nothing": iou_cf_mf, "maximized": iou_cf_mt, "corrected": iou_ct_mf, "all": iou_ct_mt})
    df.to_csv(os.path.join(REPORTS_DIR, os.path.join(REPORTS_DIR, "sfm_mvs_results.csv")))

    click.echo(f"Generating Charts")
    SFM_PATH = os.path.join(REPORTS_DIR, "figures", "sfm")

    fig = px.box(df, points="all", title="SfM na zbiorze MVS - różne wersje modeli", )
    fig.update_layout(xaxis_title="Wersja modeli", yaxis_title="IoU", showlegend=False)
    fig.write_image(os.path.join(SFM_PATH, "sfm_mvs_results_box.png"))

    df_mean = df.mean()
    fig = px.bar(df_mean)
    fig.update_layout(xaxis_title="Wersja modeli", yaxis_title="Średnie IoU", showlegend=False)
    fig.write_image(os.path.join(SFM_PATH, "sfm_mvs_results_bar.png"))



if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter