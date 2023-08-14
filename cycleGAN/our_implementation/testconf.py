import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/content/drive/MyDrive/happysaddata"
#VAL_DIR = "/content/drive/MyDrive/data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
# CHECKPOINT_GEN_OBJ1 = "gen_obj1.pth.tar"
# CHECKPOINT_GEN_OBJ2 = "gen_obj2.pth.tar"
# CHECKPOINT_DISC_OBJ1 = "disc_obj1.pth.tar"
# CHECKPOINT_DISC_OBJ2 = "disc_obj2.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
