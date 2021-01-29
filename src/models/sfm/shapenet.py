import json
import os
from typing import Dict
from settings import SHAPENET_DATASET_DIR
import re

TAXONOMY_PATH = os.path.join(SHAPENET_DATASET_DIR, "ShapeNet_taxonomy.json")


def get_taxonomy(taxonomy_path: str) -> Dict:
    with open(taxonomy_path, "r") as f:
        s = f.read()
        taxonomies = json.loads(re.sub("//.*", "", s, flags=re.MULTILINE))
        return taxonomies



# def calc_iou(test_iou, taxonomy_id, binvox_truth_path, binvox_result_path):
#     result_vox: np.ndarray = br.read_as_3d_array(open(binvox_result_path, "rb")).data
#     truth_vox: np.ndarray = br.read_as_3d_array(open(binvox_truth_path, "rb")).data

#     iou = get_iou(result_vox, truth_vox)

#     # IoU per taxonomy
#     if taxonomy_id not in test_iou:
#         test_iou[taxonomy_id] = {"n_samples": 0, "iou": []}
#     test_iou[taxonomy_id]["n_samples"] += 1
#     test_iou[taxonomy_id]["iou"].append(iou)

#     return iou


# def run_shapenet(taxonomies):
#     test_iou = dict()

#     for taxonomy in taxonomies:
#         taxonomy_id = taxonomy["taxonomy_id"]
#         test_data = taxonomy["test"]
#         for test_id in test_data:
#             print(f"=========== TAXONOMY {taxonomy_id}: {test_id} ============")

#             data_dir = os.path.join(
#                 SHAPENET_DATASET_DIR,
#                 "ShapeNetRendering",
#                 taxonomy_id,
#                 test_id,
#                 "rendering",
#             )
#             resized_dir = os.path.join(data_dir, "resized")
#             result_dir = os.path.join(
#                 SHAPENET_DATASET_DIR, "results", "sfm", taxonomy_id, test_id
#             )
#             try:
#                 subprocess.run(["pwsh", "-command", f"rm {resized_dir} -Recurse"])
#             except Exception as e:
#                 print(e)

#             # try:
#             #     subprocess.run(["pwsh", "-command", f"rm {result_dir} -Recurse"])
#             # except Exception as e:
#             #     print(e)

#             subprocess.run(["pwsh", "-command", f"mkdir {resized_dir}"])
#             subprocess.run(
#                 f"magick mogrify -resize 1024x1024 -path {resized_dir} {os.path.join(data_dir, '*.png')}"
#             )

#             command = f"docker run -v {DATA_DIR}:/data --user 0 --rm -it spedenaave/dpg pipeline.py --input /data/ShapeNet/ShapeNetRendering/{taxonomy_id}/{test_id}/rendering/resized --output /data/ShapeNet/results/sfm/{taxonomy_id}/{test_id} --sfm-type global --flength 400 --geomodel e --run-openmvg --run-openmvs --dpreset ULTRA --descmethod AKAZE_FLOAT --rmcuda --output-obj --densify-only"
#             subprocess.run(command)

#             ply_path = os.path.join(result_dir, "omvs", "scene_dense.ply")
#             binvox_result_path = os.path.join(result_dir, "omvs", "scene_dense.binvox")
#             binvox_truth_path = os.path.join(
#                 SHAPENET_DATASET_DIR,
#                 "ShapeNetVox32",
#                 taxonomy_id,
#                 test_id,
#                 "model.binvox",
#             )
#             try:
#                 plyToBinvox(ply_path, binvox_result_path)
#                 iou = calc_iou(
#                     test_iou, taxonomy_id, binvox_truth_path, binvox_result_path
#                 )
#                 print(
#                     f"\n\n==================IOU {taxonomy_id} - {test_id}: {iou}======================\n\n"
#                 )
#             except Exception as e:
#                 print(e)
#     return test_iou

# def show_binvoxes(taxonomies):
#     test_iou = dict()
#     for taxonomy in taxonomies:
#         taxonomy_id = taxonomy["taxonomy_id"]
#         test_data = taxonomy["test"]
#         for test_id in test_data:
#             print(f"=========== TAXONOMY {taxonomy_id}: {test_id} ============")

#             result_dir = os.path.join(
#                 SHAPENET_DATASET_DIR, "results", "sfm", taxonomy_id, test_id
#             )
#             binvox_result_path = os.path.join(result_dir, "omvs", "scene_dense.binvox")
#             binvox_truth_path = os.path.join(
#                 SHAPENET_DATASET_DIR,
#                 "ShapeNetVox32",
#                 taxonomy_id,
#                 test_id,
#                 "model.binvox",
#             )
#             if os.path.exists(binvox_result_path):
#                 subprocess.run([VIEWVOX_EXE, binvox_truth_path])
#                 try:
#                     subprocess.run([VIEWVOX_EXE, binvox_result_path])
#                     iou = calc_iou(
#                         test_iou, taxonomy_id, binvox_truth_path, binvox_result_path
#                     )
#                     print("IOU:", iou)
#                 except Exception as e:
#                     print(e)
#             else:
#                 print(f"{taxonomy_id}: {test_id} - result doesn't exist")
#     print(test_iou)