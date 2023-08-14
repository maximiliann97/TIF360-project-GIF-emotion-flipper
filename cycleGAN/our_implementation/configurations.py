import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/content/drive/MyDrive/happysaddata"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
object_1 = 'cropped_happy_detected_only'
object_2 = 'cropped_sad_detected_only'

transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])