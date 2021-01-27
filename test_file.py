from models.model_types import Pix2VoxTypes
from src.models.Pix2Vox.runner import  train_model, test_model
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train_models():
    shapenet_undersampled_10_times = 10
    shapenet_undersampled_50_times = 50

    train_model(Pix2VoxTypes.Pix2Vox_Plus_Plus_F, "Mixed", "Mixed", shapenet_undersampled_10_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")
    train_model(Pix2VoxTypes.Pix2Vox_Plus_Plus_A, "Mixed", "Mixed", shapenet_undersampled_10_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")
    train_model(Pix2VoxTypes.Pix2Vox_F, "Mixed", "Mixed", shapenet_undersampled_10_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")
    train_model(Pix2VoxTypes.Pix2Vox_A, "Mixed", "Mixed", shapenet_undersampled_10_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")

    train_model(Pix2VoxTypes.Pix2Vox_Plus_Plus_F, "Mixed", "Mixed", shapenet_undersampled_50_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")
    train_model(Pix2VoxTypes.Pix2Vox_Plus_Plus_A, "Mixed", "Mixed", shapenet_undersampled_50_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")
    train_model(Pix2VoxTypes.Pix2Vox_F, "Mixed", "Mixed", shapenet_undersampled_50_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")
    train_model(Pix2VoxTypes.Pix2Vox_A, "Mixed", "Mixed", shapenet_undersampled_50_times, 8,
         "data/mvs_dataset/MVS_taxonomy_for_training.json")


def test_models():
    mvs_full_taxonomy_path = "data/mvs_dataset/MVS_taxonomy.json"
    mvs_train_taxonomy_path = "data/mvs_dataset/MVS_taxonomy_for_training.json"

    model_types_and_paths_only_shapenet = [(Pix2VoxTypes.Pix2Vox_Plus_Plus_F,  "models/Pix2Vox++-F-ShapeNet.pth"),
                             (Pix2VoxTypes.Pix2Vox_Plus_Plus_A,  "models/Pix2Vox++-A-ShapeNet.pth"),
                             (Pix2VoxTypes.Pix2Vox_F,  "models/Pix2Vox-F-ShapeNet.pth"),
                             (Pix2VoxTypes.Pix2Vox_A,  "models/Pix2Vox-A-ShapeNet.pth")]


    model_types_and_paths_shapenet_and_mvs = [(Pix2VoxTypes.Pix2Vox_Plus_Plus_F,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_F_Mixed_10/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_Plus_Plus_A,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_A_Mixed_10/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_F,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_F_Mixed_10/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_A,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_A_Mixed_10/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_Plus_Plus_F,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_F_Mixed_50/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_Plus_Plus_A,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_A_Mixed_50/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_F,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_F_Mixed_50/checkpoint-best.pth"),
                             (Pix2VoxTypes.Pix2Vox_A,  "output/checkpoints_Pix2VoxTypes.Pix2Vox_A_Mixed_50/checkpoint-best.pth")]

    n_views = [1, 5, 10, 20, 30]
    for model_type, weights_path in model_types_and_paths_only_shapenet:
        file_name = weights_path.split("/")[-1]
        model_name = file_name[:file_name.index(".pth")]

        if not os.path.exists(f"reports/results/ShapeNet_{model_name}_1.csv"):
            test_model(model_type, "ShapeNet", 8, mvs_full_taxonomy_path, weights_path=weights_path, n_views=1,
                       results_file_name=f"reports/results/ShapeNet_{model_name}_1.csv")

        for n_view in n_views:
            if not os.path.exists(f"reports/results/MVS_{model_name}_{n_view}.csv"):
                test_model(model_type,"MVS", 8, mvs_full_taxonomy_path, weights_path=weights_path, n_views=n_view, results_file_name = f"reports/results/MVS_{model_name}_{n_view}.csv")

    for model_type, weights_path in model_types_and_paths_shapenet_and_mvs:
        directory_name = weights_path.split("/")[1]
        model_name = directory_name[directory_name.index("Pix2VoxTypes")+13:]
        test_model(model_type, "ShapeNet", 8, mvs_train_taxonomy_path, weights_path=weights_path, n_views=1,
                   results_file_name=f"reports/results/ShapeNet_{model_name}_1.csv")
        for n_view in n_views:
            test_model(model_type,"MVS", 8, mvs_train_taxonomy_path, weights_path=weights_path, n_views=n_view, results_file_name = f"reports/results/MVS_{model_name}_{n_view}.csv")


if __name__ == '__main__':
    train_models()
    test_models()