import os
import subprocess
from settings import DATA_DIR, MVS_DATASET_DIR


def get_mvs_img_path(scan_id: int, image_name: str = "clean_032_max.png"):
    return os.path.join(MVS_DATASET_DIR, "images", f"scan{scan_id}", image_name)


def get_mvs_result_ply_path(scan_id: int, corrected: bool):
    return os.path.join(
        MVS_DATASET_DIR,
        "results",
        "sfm",
        f"scan{scan_id}",
        "omvs",
        f"scene_dense{'_corrected' if corrected else ''}.ply",
    )


def get_mvs_result_vox_path(scan_id: int, corrected: bool, maximized: bool = False):
    return os.path.join(
        MVS_DATASET_DIR,
        "results",
        "sfm",
        f"scan{scan_id}",
        "omvs",
        f"scene_dense{'_corrected' if corrected else ''}{'_maximized' if maximized else ''}.binvox",
    )


def get_mvs_truth_ply_path(scan_id: int, corrected: bool):
    return os.path.join(MVS_DATASET_DIR, "points_clouds", f"stl{scan_id:03d}_total{'_corrected' if corrected else ''}.ply")


def get_mvs_truth_vox_path(scan_id: int, corrected: bool):
    return os.path.join(MVS_DATASET_DIR, "voxels", f"stl{scan_id:03d}_total{'_corrected' if corrected else ''}.binvox")


def run_mvs_reconstruction(scan_id: int):
    command = f"docker run -v {DATA_DIR}:/data --user 0 --rm -it spedenaave/dpg pipeline.py --input /data/mvs_dataset/images/scan{scan_id} --output /data/mvs_dataset/results/sfm/scan{scan_id} --sfm-type global --flength 1920 --geomodel e --run-openmvg --run-openmvs --rmcuda --output-obj --densify-only"
    try:
        subprocess.run(command)
    except:
        print(f"\nERROR: MVS RECONSTRUCTION {scan_id}")


def run_mvs_position_correction(scan_id: int, cloud_compare_path: str, corrected: bool):
    command = f"{cloud_compare_path} {get_mvs_result_ply_path(scan_id, corrected)} {get_mvs_truth_ply_path(scan_id,corrected)}"
    try:
        subprocess.run(command)
    except:
        print(f"\nERROR: MVS POSITION CORRECTION {scan_id}")

