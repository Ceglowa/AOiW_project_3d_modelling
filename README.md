AOiW - reconstruction of 3d models from sequence of images
==============================

The goal of the project was to tackle the problem of reconstructing 3d models from sequence 
 of images of given object. We implemented two solutions: one using classical method - `Structure
 From Motion` and the other using deep neural network - `Pix2Vox`.

## How to prepare `data` folder for experiments

For experiments to start, you need to make sure your `data` includes all the necessary data. In `data` 
folder there is a file called `where_to_download_data.txt`, which includes all needed links and 
describes how you should prepare your `data` folder.
To summarize here how the `data` folder should look like, here is a description:

```
├── data
│   ├── mvs_dataset       <- MVS data.
│   │   ├── images        <- Folder for all images of the MVS dataset. They are included in cleaned_images.zip
│   │   ├── point_clouds  <- @TODO KAcper Tu mają być oryginalne point cloudy i twoje corrected
│   │   ├── @TODO Kolejne foldery dla SfM
│   │   ├── processed_voxels_pix2vox <- Folder with processed voxels which are needed for Pix2Vox model. They are stored in processed_voxels_pix2vox.zip
│   │   ├──
│   ├── ShapeNet        <- ShapeNet data.
│   │   ├── ShapeNetRendering <- Images for ShapeNet dataset. Included in ShapeNetRendering.tgz
│   │   └── ShapeNetVox32     <- Voxels for objects in ShapeNet dataset. Included in ShapeNetVox32.tgz
```


## How to run experiments for `Structure From Motion`

## How to run experiments for `Pix2Vox`

Make sure you have 4 pretrained models of `Pix2Vox` in `models` directory. In this
directory there is a `.txt` with links to those models. Here we are also posting these links:
1. https://gateway.infinitescript.com/?fileName=Pix2Vox-A-ShapeNet.pth - Pix2Vox-A
1. https://gateway.infinitescript.com/?fileName=Pix2Vox-F-ShapeNet.pth - Pix2Vox-F
1. https://gateway.infinitescript.com/?fileName=Pix2Vox%2B%2B-A-ShapeNet.pth - Pix2Vox++ A
1. https://gateway.infinitescript.com/?fileName=Pix2Vox%2B%2B-F-ShapeNet.pth - Pix2Vox++ F

To run training for the new models, run `train_pix2vox_models_and_test.sh` shell script.
This script trains new models and also tests them.

To use the trained models and run only tests, run `test_pix2vox_models.sh` shell script.

To visualize results generated by testing script, run `visualize_pix2vox_results.sh` shell script.
