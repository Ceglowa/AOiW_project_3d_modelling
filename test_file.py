from models.model_types import Pix2VoxTypes
from src.models.Pix2Vox.runner import main

if __name__ == '__main__':
    main(True, Pix2VoxTypes.Pix2Vox_Plus_Plus_A,  None)
    # run(Pix2VoxTypes.Pix2Vox_Plus_Plus_A, "models/Pix2Vox++-A-ShapeNet.pth", "data/mvs_dataset/images/scan1")
