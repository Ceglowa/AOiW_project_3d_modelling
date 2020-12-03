import os

PROJECT_DIR = os.path.normpath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MVS_DATASET_DIR = os.path.join(DATA_DIR, "mvs_dataset")
SHAPENET_DATASET_DIR = os.path.join(DATA_DIR, "ShapeNet")

PROGRAMS_DIR = os.path.join(PROJECT_DIR, "programs")
VIEWVOX_EXE = os.path.join(PROGRAMS_DIR, "viewvox.exe")