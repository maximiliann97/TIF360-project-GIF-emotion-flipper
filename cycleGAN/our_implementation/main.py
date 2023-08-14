from train import Trainer
from models import Discriminator, Generator
from torch.utils.data import DataLoader
from dataset import Object1Object2Dataset
import torch.optim as optim
import configurations
import torch
import torch.nn as nn



def main(load_checkpoint):

    # Initialize
    disc_obj1 = Discriminator(in_channels=3).to(configurations.DEVICE)
    disc_obj2 = Discriminator(in_channels=3).to(configurations.DEVICE)
    gen_obj1 = Generator(img_channels=3, num_residuals=9).to(configurations.DEVICE)
    gen_obj2 = Generator(img_channels=3, num_residuals=9).to(configurations.DEVICE)
    opt_disc = optim.Adam(
        list(disc_obj1.parameters()) + list(disc_obj2.parameters()),
        lr=configurations.LEARNING_RATE,
        betas=(0.5, 0.999),
        )

    opt_gen = optim.Adam(
        list(gen_obj1.parameters()) + list(gen_obj2.parameters()),
        lr=configurations.LEARNING_RATE,
        betas=(0.5, 0.999),
        )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    dataset = Object1Object2Dataset(
        root_obj1=configurations.DATA_DIR + '/' + configurations.object_1,
        root_obj2=configurations.DATA_DIR + '/' + configurations.object_2,
        transform=configurations.transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=configurations.BATCH_SIZE,
        shuffle=True,
        num_workers=configurations.NUM_WORKERS,
        pin_memory=True,
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    trainer = Trainer(disc_obj1, disc_obj2, gen_obj1, gen_obj2, loader, opt_disc, opt_gen, d_scaler, g_scaler, L1, mse)


    if load_checkpoint:
        # Load checkpoints
        gen_obj1_checkpoint = "gen_obj1_checkpoint.pth"
        gen_obj2_checkpoint = "gen_obj2_checkpoint.pth"
        disc_obj1_checkpoint = "disc_obj1_checkpoint.pth"
        disc_obj2_checkpoint = "disc_obj2_checkpoint.pth"
        trainer.load_checkpoint(trainer.gen_obj1, trainer.opt_gen, gen_obj1_checkpoint)
        trainer.load_checkpoint(trainer.gen_obj2, trainer.opt_gen, gen_obj2_checkpoint)
        trainer.load_checkpoint(trainer.disc_obj1, trainer.opt_disc, disc_obj1_checkpoint)
        trainer.load_checkpoint(trainer.disc_obj2, trainer.opt_disc, disc_obj2_checkpoint)

    # Training
    trainer.train(configurations.object_1, configurations.object_2, configurations.NUM_EPOCHS)


if __name__== "__main__":
    main(configurations.LOAD_MODEL)








