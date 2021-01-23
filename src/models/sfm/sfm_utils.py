from itertools import product
import os
import subprocess
from settings import VIEWVOX_EXE
from typing import Callable, Tuple
from pyntcloud import PyntCloud
import utils.binvox_rw as br
import numpy as np


CLOUDCOMPARE_PATH = os.getenv("CLOUD_COMPARE_PATH")


def convertPlyToBinvox(cloud: PyntCloud) -> br.Voxels:
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32)
    voxelgrid = cloud.structures[voxelgrid_id]
    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((32, 32, 32)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True

    return br.Voxels(voxel, (32, 32, 32), (0, 0, 0), 1, "xyz")


def readAndSavePlyToBinvox(input_path: str, output_path: str) -> br.Voxels:
    cloud = PyntCloud.from_file(input_path)
    voxels = convertPlyToBinvox(cloud)
    with open(output_path, "wb") as f:
        voxels.write(f)

    return voxels


def get_iou(result_vox_data: np.ndarray, truth_vox_data: np.ndarray) -> float:
    intersection = np.sum(result_vox_data * truth_vox_data)
    union = np.sum((result_vox_data + truth_vox_data) >= 1)
    iou = intersection / union

    return iou


def get_maximized_result_vox_data(
    result_vox_data: np.ndarray,
    truth_vox_data: np.ndarray,
    max_shift: int = 10,
    comp_func: Callable[[np.ndarray, np.ndarray], float] = get_iou,
) -> Tuple[float, np.ndarray, Tuple]:
    shifts = product(range(-max_shift, max_shift + 1), repeat=3)
    maximized_result_vox_data: np.ndarray = None  # type: ignore
    max_iou = 0
    best_shift: Tuple = None  # type: ignore
    for shift in shifts:
        result_vox_shifted_data: np.ndarray = np.roll(result_vox_data, shift)  # type: ignore
        iou = comp_func(result_vox_shifted_data, truth_vox_data)
        if iou > max_iou:
            max_iou = iou
            maximized_result_vox_data = result_vox_shifted_data
            best_shift = shift

    return max_iou, maximized_result_vox_data, best_shift


def view_voxel(voxel_path: str):
    if os.path.exists(voxel_path):
        subprocess.run([VIEWVOX_EXE, voxel_path])


def read_voxel(voxel_path: str) -> br.Voxels:
    with open(voxel_path, "rb") as f:
        return br.read_as_3d_array(f)
