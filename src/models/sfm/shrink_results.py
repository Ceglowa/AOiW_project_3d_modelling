import os
import shutil
from settings import MVS_DATASET_DIR
from tqdm.auto import tqdm
from sfm_utils import is_correct_scan_id

def main():
    for i in tqdm(range(78, 129)):
        if is_correct_scan_id(i):
            original_path = os.path.join(MVS_DATASET_DIR, "results", "sfm", f"scan{i}", "omvs")
            original_cloud = os.path.join(original_path, "scene_dense.ply")
            original_cloud_corrected = os.path.join(original_path, "scene_dense_corrected.ply")

            new_path = os.path.join(MVS_DATASET_DIR, "results_new", "sfm", f"scan{i}", "omvs")
            new_cloud = os.path.join(new_path, "scene_dense.ply")
            new_cloud_corrected = os.path.join(new_path, "scene_dense_corrected.ply")

            try:
                shutil.copy(original_cloud, new_cloud)
            except IOError as io_err:
                os.makedirs(os.path.dirname(new_cloud))
                shutil.copy(original_cloud, new_cloud)

            shutil.copy(original_cloud_corrected, new_cloud_corrected)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter


