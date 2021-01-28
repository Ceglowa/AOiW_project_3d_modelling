import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

t_list = ['0.2', '0.3', '0.4', '0.5']
path_to_plots = "../../reports/figures/pix2vox_plots"

def visualize_shapenet_results(path_to_results: str, files: List[str], train_data_name: str, substr_to_find: str):
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(path_to_results, f))
        ious = []
        ts = []
        for t in t_list:
            t_results = list(df[t])
            ious.extend(t_results)
            ts.extend([t for _ in range(len(t_results))])

        new_df = pd.DataFrame(data={"IoU": ious,
                                    "t": ts})
        new_df['Architecture'] = f[len("ShapeNet") + 1:f.index(substr_to_find)]
        dfs.append(new_df)

    full_df = pd.concat(objs=dfs)
    plt.figure(figsize=(7,4))
    sns.boxplot(data=full_df, hue='t', x='Architecture', y='IoU')
    plt.title(f"Reconstruction results on test ShapeNet for models trained on {train_data_name}")
    plt.savefig(os.path.join(path_to_plots, f"ShapeNet_test_results_trained_on_{train_data_name}.png"))
    plt.close()

def visualize_mvs_result_for_model(path_to_results: str, files: List[str], model_name: str, mvs_dataset_type: str,
                                   ylim: Tuple[float, float],
                                   train_data_name: str):
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(path_to_results, f))
        ious = []
        ts = []
        for t in t_list:
            t_results = list(df[t])
            ious.extend(t_results)
            ts.extend([t for _ in range(len(t_results))])

        new_df = pd.DataFrame(data={"IoU": ious,
                                    "t": ts})
        new_df['n_views'] = int(f.split(".")[0].split("_")[-1])
        dfs.append(new_df)

    full_df = pd.concat(objs=dfs)

    sns.lineplot(data=full_df, x='n_views', y='IoU', hue='t')
    plt.title(f"Reconstruction results on {mvs_dataset_type} for {model_name} trained on {train_data_name}")
    plt.ylim(ylim)
    plt.savefig(os.path.join(path_to_plots, f"{mvs_dataset_type}_results_{model_name}_trained_on_{train_data_name}.png"))
    plt.close()

def visualize_mvs_results(path_to_results: str,
                                              pix2vox_a_files: List[str], pix2vox_f_files: List[str],
                                              pix2voxpp_a_files: List[str],
                                              pix2voxpp_f_files: List[str], dataset_name: str, test_dataset_name: str, ylim: Tuple[float, float]):
    visualize_mvs_result_for_model(path_to_results, pix2vox_a_files, "Pix2Vox-A", test_dataset_name, ylim, dataset_name)
    visualize_mvs_result_for_model(path_to_results, pix2vox_f_files, "Pix2Vox-F", test_dataset_name, ylim, dataset_name)
    visualize_mvs_result_for_model(path_to_results, pix2voxpp_a_files, "Pix2Vox++A", test_dataset_name, ylim,
                                   dataset_name)
    visualize_mvs_result_for_model(path_to_results, pix2voxpp_f_files, "Pix2Vox++F", test_dataset_name, ylim,
                                   dataset_name)

def visualize_processing_time(path_to_results: str, files: List[str]):
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(path_to_results, f))
        dfs.append(df)

    full_df = pd.concat(objs=dfs)

    sns.lineplot(data=full_df, x='n_views', y='time')
    plt.ylabel("Processing time for one model (in ms)")
    plt.xlabel("Number of images")
    plt.title(f"Processing times in relation to number of images for one object")
    plt.savefig(os.path.join(path_to_plots, "mvs_images_processing_times.png"))
    plt.close()


if __name__ == '__main__':
    path_to_results = os.path.join("..", "..", "reports", "results")
    all_results = os.listdir(path_to_results)

    # visualize_shapenet_results(path_to_results, list(
    #     filter(lambda s: s.startswith("ShapeNet_Pix2Vox") and "-ShapeNet" in s, all_results)), "ShapeNet", "-ShapeNet")
    # visualize_shapenet_results(path_to_results, list(
    #     filter(lambda s: s.startswith("ShapeNet_Pix2Vox") and "Mixed_10" in s, all_results)), "Mixed_10", "_Mixed_10")
    # visualize_shapenet_results(path_to_results, list(
    #     filter(lambda s: s.startswith("ShapeNet_Pix2Vox") and "Mixed_50" in s, all_results)), "Mixed_50", "_Mixed_50")
    # visualize_mvs_results(path_to_results,
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox-A-ShapeNet"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox-F-ShapeNet"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox++-A-ShapeNet"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox++-F-ShapeNet"),
    #                                                       all_results)),
    #                                           "ShapeNet", "Full MVS", (0.02, 0.12))
    #
    # visualize_mvs_results(path_to_results,
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_A_Mixed_10"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_F_Mixed_10"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_Plus_Plus_A_Mixed_10"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_Plus_Plus_F_Mixed_10"),
    #                                                       all_results)),
    #                                           "Mixed_10", "Test MVS", (0.01, 0.35))
    #
    # visualize_mvs_results(path_to_results,
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_A_Mixed_50"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_F_Mixed_50"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_Plus_Plus_A_Mixed_50"),
    #                                                       all_results)),
    #                                           list(filter(lambda s: s.startswith("MVS_Pix2Vox_Plus_Plus_F_Mixed_50"),
    #                                                       all_results)),
    #                                           "Mixed_50", "Test MVS", (0.01, 0.35))

    visualize_processing_time(path_to_results, list(filter(lambda s: s.startswith("MVS_time_processing"),
                                                          all_results)))