import click
import os
from src.models.Pix2Vox.models.model_types import Pix2VoxTypes
from src.models.Pix2Vox.runner import train_model, test_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_models(path_to_mvs_dataset: str):
    shapenet_undersampled_10_times = 10
    shapenet_undersampled_50_times = 50
    batch_size = 8

    for model in [Pix2VoxTypes.Pix2Vox_Plus_Plus_F, Pix2VoxTypes.Pix2Vox_Plus_Plus_A, Pix2VoxTypes.Pix2Vox_F,
                  Pix2VoxTypes.Pix2Vox_A]:
        train_model(model, "Mixed", "Mixed", shapenet_undersampled_10_times, batch_size,
                    os.path.join(path_to_mvs_dataset, "MVS_taxonomy_for_training.json"))
        train_model(model, "Mixed", "Mixed", shapenet_undersampled_50_times, batch_size,
                    os.path.join(path_to_mvs_dataset, "MVS_taxonomy_for_training.json"))


def test_models(path_to_mvs_dataset: str, path_to_models: str, path_to_outputs: str, path_to_results: str):
    mvs_full_taxonomy_path = os.path.join(path_to_mvs_dataset, "MVS_taxonomy.json")
    mvs_train_taxonomy_path = os.path.join(path_to_mvs_dataset, "MVS_taxonomy_for_training.json")

    model_types_and_paths_only_shapenet = [
        (Pix2VoxTypes.Pix2Vox_Plus_Plus_F, os.path.join(path_to_models, "Pix2Vox++-F-ShapeNet.pth")),
        (Pix2VoxTypes.Pix2Vox_Plus_Plus_A, os.path.join(path_to_models, "Pix2Vox++-A-ShapeNet.pth")),
        (Pix2VoxTypes.Pix2Vox_F, os.path.join(path_to_models, "Pix2Vox-F-ShapeNet.pth")),
        (Pix2VoxTypes.Pix2Vox_A, os.path.join(path_to_models, "Pix2Vox-A-ShapeNet.pth"))]

    model_types_and_paths_shapenet_and_mvs = [(Pix2VoxTypes.Pix2Vox_Plus_Plus_F,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_F_Mixed_10/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_Plus_Plus_A,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_A_Mixed_10/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_F,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_F_Mixed_10/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_A,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_A_Mixed_10/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_Plus_Plus_F,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_F_Mixed_50/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_Plus_Plus_A,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_Plus_Plus_A_Mixed_50/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_F,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_F_Mixed_50/checkpoint-best.pth")),
                                              (Pix2VoxTypes.Pix2Vox_A,
                                               os.path.join(path_to_outputs,
                                                            "checkpoints_Pix2VoxTypes.Pix2Vox_A_Mixed_50/checkpoint-best.pth"))]

    n_views = [1, 5, 10, 20, 30]
    for model_type, weights_path in model_types_and_paths_only_shapenet:
        file_name = weights_path.split("/")[-1]
        model_name = file_name[:file_name.index(".pth")]

        test_model(model_type, "ShapeNet", 8, mvs_full_taxonomy_path, weights_path=weights_path, n_views=1,
                   results_file_name=os.path.join(path_to_results, f"ShapeNet_{model_name}_1.csv"))
        for n_view in n_views:
            test_model(model_type, "MVS", 8, mvs_full_taxonomy_path, weights_path=weights_path, n_views=n_view,
                       results_file_name=os.path.join(path_to_results, f"MVS_{model_name}_{n_view}.csv"))

    for model_type, weights_path in model_types_and_paths_shapenet_and_mvs:
        directory_name = weights_path.split("/")[1]
        model_name = directory_name[directory_name.index("Pix2VoxTypes") + 13:]
        test_model(model_type, "ShapeNet", 8, mvs_train_taxonomy_path, weights_path=weights_path, n_views=1,
                   results_file_name=os.path.join(path_to_results,f"ShapeNet_{model_name}_1.csv"))
        for n_view in n_views:
            test_model(model_type, "MVS", 8, mvs_train_taxonomy_path, weights_path=weights_path, n_views=n_view,
                       results_file_name=os.path.join(path_to_results,f"MVS_{model_name}_{n_view}.csv"))


def show_best_voxels(path_to_mvs_dataset: str, path_to_outputs: str):
    test_model(Pix2VoxTypes.Pix2Vox_A, "MVS", 8, os.path.join(path_to_mvs_dataset, "MVS_taxonomy_best.json"),
               weights_path=os.path.join(path_to_outputs,"checkpoints_Pix2VoxTypes.Pix2Vox_A_Mixed_10/checkpoint-best.pth"),
               n_views=30, save_results_to_file=False, show_voxels=True)


def test_show_times(path_to_mvs_dataset: str, path_to_outputs: str, path_to_results_dir: str):
    for n_views in [1, 5, 10, 15, 20, 25, 30, 35, 40]:
        test_model(Pix2VoxTypes.Pix2Vox_A, "MVS", 8, os.path.join(path_to_mvs_dataset,"MVS_taxonomy.json"),
                   weights_path=os.path.join(path_to_outputs, "checkpoints_Pix2VoxTypes.Pix2Vox_A_Mixed_10/checkpoint-best.pth"),
                   n_views=n_views, save_results_to_file=False, show_voxels=False,
                   path_to_times_csv=os.path.join(path_to_results_dir,f"MVS_time_processing_n_views_{n_views}.csv"))


@click.command()
@click.option(
    "-tr",
    "--run-train",
    "run_train",
    type=bool,
    required=True,
    default=True
)
@click.option(
    "-d",
    "--path-to-mvs-dataset",
    "path_to_mvs_dataset",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
@click.option(
    "-r",
    "--path-to-results-dir",
    "path_to_results_dir",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
@click.option(
    "-m",
    "--path-to-models",
    "path_to_models",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
@click.option(
    "-o",
    "--path-to-outputs",
    "path_to_outputs",
    type=click.Path(dir_okay=True, exists=True),
    required=True
)
def run_experiments(run_train: bool, path_to_mvs_dataset: str, path_to_results_dir: str, path_to_models: str, path_to_outputs: str):
    if run_train:
        train_models(path_to_mvs_dataset)
    # test_models(path_to_mvs_dataset, path_to_models, path_to_outputs, path_to_results_dir)
    show_best_voxels(path_to_mvs_dataset, path_to_outputs)
    test_show_times(path_to_mvs_dataset, path_to_outputs, path_to_results_dir)


if __name__ == '__main__':
    run_experiments()
