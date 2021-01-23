from mvs import get_mvs_result_vox_path, get_mvs_truth_vox_path
from settings import *
from sfm_utils import get_iou, get_maximized_result_vox_data, read_voxel, view_voxel


def main():
    truth = read_voxel(get_mvs_truth_vox_path(1, corrected=True))
    result = read_voxel(get_mvs_result_vox_path(1, corrected=True))

    print(get_maximized_result_vox_data(result.data, truth.data))
    print(get_iou(result.data, truth.data))
    view_voxel(get_mvs_truth_vox_path(1, corrected=True))
    view_voxel(get_mvs_result_vox_path(1, corrected=True))


if __name__ == "__main__":
    main()