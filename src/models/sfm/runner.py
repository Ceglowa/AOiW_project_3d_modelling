import subprocess
from settings import *
import json
import numpy as np
import utils.binvox_rw as br
from pyntcloud import PyntCloud
import re
import cv2 as cv


def plyToBinvox(input_path, output_path):
    cloud = PyntCloud.from_file(input_path)

    # cloud.plot(mesh=True, backend="threejs")

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32)
    voxelgrid = cloud.structures[voxelgrid_id]
    # voxelgrid.plot(d=3, mode="density", cmap="hsv")

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((32, 32, 32)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True

    with open(output_path, 'wb') as f:
        v = br.Voxels(voxel, (32, 32, 32), (0, 0, 0), 1, 'xyz')
        v.write(f)

    return v


def calc_iou(test_iou, taxonomy_id, binvox_truth_path, binvox_result_path):
    result_vox: np.ndarray = br.read_as_3d_array(open(binvox_result_path, "rb")).data
    truth_vox: np.ndarray = br.read_as_3d_array(open(binvox_truth_path, "rb")).data

    intersection = np.sum(result_vox * truth_vox)
    union = np.sum((result_vox + truth_vox) >= 1)
    iou = intersection / union

    # IoU per taxonomy
    if taxonomy_id not in test_iou:
        test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
    test_iou[taxonomy_id]['n_samples'] += 1
    test_iou[taxonomy_id]['iou'].append(iou)

    return iou

def run_shapenet(taxonomies):
    test_iou = dict()

    for taxonomy in taxonomies:
        taxonomy_id = taxonomy["taxonomy_id"]
        test_data = taxonomy["test"]
        for test_id in test_data:
            print(f"=========== TAXONOMY {taxonomy_id}: {test_id} ============")

            data_dir = os.path.join(SHAPENET_DATASET_DIR, "ShapeNetRendering", taxonomy_id, test_id, "rendering")
            resized_dir = os.path.join(data_dir, "resized")
            result_dir = os.path.join(SHAPENET_DATASET_DIR, "results", "sfm", taxonomy_id, test_id)
            try:
                subprocess.run(["pwsh", "-command", f"rm {resized_dir} -Recurse"])
            except Exception as e:
                print(e)

            # try:
            #     subprocess.run(["pwsh", "-command", f"rm {result_dir} -Recurse"])
            # except Exception as e:
            #     print(e)

            subprocess.run(["pwsh", "-command", f"mkdir {resized_dir}"])
            subprocess.run(f"magick mogrify -resize 1024x1024 -path {resized_dir} {os.path.join(data_dir, '*.png')}")

            command = f"docker run -v {DATA_DIR}:/data --user 0 --rm -it spedenaave/dpg pipeline.py --input /data/ShapeNet/ShapeNetRendering/{taxonomy_id}/{test_id}/rendering/resized --output /data/ShapeNet/results/sfm/{taxonomy_id}/{test_id} --sfm-type global --flength 400 --geomodel e --run-openmvg --run-openmvs --dpreset ULTRA --descmethod AKAZE_FLOAT --rmcuda --output-obj --densify-only"
            subprocess.run(command)

            ply_path = os.path.join(result_dir, "omvs", "scene_dense.ply")
            binvox_result_path = os.path.join(result_dir, "omvs", "scene_dense.binvox")
            binvox_truth_path = os.path.join(SHAPENET_DATASET_DIR, "ShapeNetVox32", taxonomy_id, test_id, "model.binvox")
            try:
                plyToBinvox(ply_path, binvox_result_path)
                iou = calc_iou(test_iou, taxonomy_id, binvox_truth_path, binvox_result_path)
                print(f"\n\n==================IOU {taxonomy_id} - {test_id}: {iou}======================\n\n")
            except Exception as e:
                print(e)
    return test_iou

            
def show_binvoxes(taxonomies):
    test_iou = dict()
    for taxonomy in taxonomies:
        taxonomy_id = taxonomy["taxonomy_id"]
        test_data = taxonomy["test"]
        for test_id in test_data:
            print(f"=========== TAXONOMY {taxonomy_id}: {test_id} ============")

            result_dir = os.path.join(SHAPENET_DATASET_DIR, "results", "sfm", taxonomy_id, test_id)
            binvox_result_path = os.path.join(result_dir, "omvs", "scene_dense.binvox")
            binvox_truth_path = os.path.join(SHAPENET_DATASET_DIR, "ShapeNetVox32", taxonomy_id, test_id, "model.binvox")
            if os.path.exists(binvox_result_path):
                subprocess.run([VIEWVOX_EXE, binvox_truth_path])
                try:
                    subprocess.run([VIEWVOX_EXE, binvox_result_path])
                    iou = calc_iou(test_iou, taxonomy_id, binvox_truth_path, binvox_result_path)
                    print("IOU:", iou)
                except Exception as e:
                    print(e)
            else:
                print(f"{taxonomy_id}: {test_id} - result doesn't exist")
    print(test_iou)


def run_mvs():
    for i in range(1, 129):
        print(f"\n\n==============SCAN {i}===============\n\n")
        command = f"docker run -v {DATA_DIR}:/data --user 0 --rm -it spedenaave/dpg pipeline.py --input /data/mvs_dataset/images/scan{i} --output /data/mvs_dataset/results/sfm/scan{i} --sfm-type global --flength 1920 --geomodel e --run-openmvg --run-openmvs --rmcuda --output-obj --densify-only"
        try:
            return_code = subprocess.run(command)   
        except:
            pass


def mvs_test(i=1):
    img_path = os.path.join(MVS_DATASET_DIR, "images", f"scan{i}", "clean_032_max.png")
    mvs_result_ply_path = os.path.join(MVS_DATASET_DIR, "results", "sfm", f"scan{i}", "omvs", "scene_dense.ply")
    mvs_result_vox_path = os.path.join(MVS_DATASET_DIR, "results", "sfm", f"scan{i}", "omvs", "scene_dense.binvox")
    mvs_truth_ply_path = os.path.join(MVS_DATASET_DIR, "points_clouds", f"stl{i:03d}_total.ply")
    mvs_truth_vox_path = os.path.join(MVS_DATASET_DIR, "voxels", f"stl{i:03d}_total.binvox")

    img = cv.imread(img_path)
    cv.imshow("img", img)
    cv.waitKey()
    cv.destroyAllWindows()

    plyToBinvox(mvs_result_ply_path, mvs_result_vox_path)
    plyToBinvox(mvs_truth_ply_path, mvs_truth_vox_path)
    subprocess.run([VIEWVOX_EXE, mvs_truth_vox_path])
    subprocess.run([VIEWVOX_EXE, mvs_result_vox_path])
    result_vox: np.ndarray = br.read_as_3d_array(open(mvs_result_vox_path, "rb")).data
    truth_vox: np.ndarray = br.read_as_3d_array(open(mvs_truth_vox_path, "rb")).data

    intersection = np.sum(result_vox * truth_vox)
    union = np.sum((result_vox + truth_vox) >= 1)
    iou = intersection / union
    print(iou)


def main():
    with open(os.path.join(SHAPENET_DATASET_DIR, "ShapeNet_taxonomy.json"), "r") as f:
        s = f.read()
        taxonomies = json.loads(re.sub("//.*","",s, flags=re.MULTILINE))
    print(taxonomies)


    # test_iou = run_shapenet(taxonomies)
    # print("\n\n\n", test_iou)

    # show_binvoxes(taxonomies)
    mvs_test(1)
    


if __name__ == "__main__":
    main()