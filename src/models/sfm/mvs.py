import os
import subprocess
from settings import DATA_DIR, MVS_DATASET_DIR


def get_mvs_img_path(scan_id: int, image_name: str = "clean_032_max.png"):
    return os.path.join(
        MVS_DATASET_DIR, "images", f"scan{scan_id}", image_name
    )


def get_mvs_maximized_result_vox_path(scan_id: int):
    return os.path.join(
        MVS_DATASET_DIR,
        "results",
        "sfm",
        f"scan{scan_id}",
        "omvs",
        "scene_dense_maximized.binvox",
    )


def get_mvs_result_ply_path(scan_id: int):
    return os.path.join(
        MVS_DATASET_DIR, "results", "sfm", f"scan{scan_id}", "omvs", "scene_dense.ply"
    )


def get_mvs_result_vox_path(scan_id: int):
    return os.path.join(
        MVS_DATASET_DIR,
        "results",
        "sfm",
        f"scan{scan_id}",
        "omvs",
        "scene_dense.binvox",
    )


def get_mvs_truth_ply_path(scan_id: int):
    return os.path.join(MVS_DATASET_DIR, "points_clouds", f"stl{scan_id:03d}_total.ply")


def get_mvs_truth_vox_path(scan_id: int):
    return os.path.join(MVS_DATASET_DIR, "voxels", f"stl{scan_id:03d}_total.binvox")


def run_mvs_reconstruction(scan_id: int):
    print(f"\n\n==============MVS RECONSTRUCTION {scan_id}===============\n\n")
    command = f"docker run -v {DATA_DIR}:/data --user 0 --rm -it spedenaave/dpg pipeline.py --input /data/mvs_dataset/images/scan{scan_id} --output /data/mvs_dataset/results/sfm/scan{i} --sfm-type global --flength 1920 --geomodel e --run-openmvg --run-openmvs --rmcuda --output-obj --densify-only"
    try:
        return_code = subprocess.run(command)
    except:
        print(f"ERROR: MVS RECONSTRUCTION {scan_id}")