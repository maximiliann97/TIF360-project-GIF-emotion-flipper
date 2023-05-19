# import torch
# from dataset import HorseZebraDataset
# import sys
# from utils import save_checkpoint, load_checkpoint
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim
# import config
# from tqdm import tqdm
# from torchvision.utils import save_image
# from discriminator_model import Discriminator
# from generator_model import Generator


# def train_fn(
#     disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
# ):
#     H_reals = 0
#     H_fakes = 0
#     loop = tqdm(loader, leave=True)

#     for idx, (zebra, horse) in enumerate(loop):
#         zebra = zebra.to(config.DEVICE)
#         horse = horse.to(config.DEVICE)

#         # Train Discriminators H and Z
#         with torch.cuda.amp.autocast():
#             fake_horse = gen_H(zebra)
#             D_H_real = disc_H(horse)
#             D_H_fake = disc_H(fake_horse.detach())
#             H_reals += D_H_real.mean().item()
#             H_fakes += D_H_fake.mean().item()
#             D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
#             D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
#             D_H_loss = D_H_real_loss + D_H_fake_loss

#             fake_zebra = gen_Z(horse)
#             D_Z_real = disc_Z(zebra)
#             D_Z_fake = disc_Z(fake_zebra.detach())
#             D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
#             D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
#             D_Z_loss = D_Z_real_loss + D_Z_fake_loss

#             # put it togethor
#             D_loss = (D_H_loss + D_Z_loss) / 2

#         opt_disc.zero_grad()
#         d_scaler.scale(D_loss).backward()
#         d_scaler.step(opt_disc)
#         d_scaler.update()

#         # Train Generators H and Z
#         with torch.cuda.amp.autocast():
#             # adversarial loss for both generators
#             D_H_fake = disc_H(fake_horse)
#             D_Z_fake = disc_Z(fake_zebra)
#             loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
#             loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

#             # cycle loss
#             cycle_zebra = gen_Z(fake_horse)
#             cycle_horse = gen_H(fake_zebra)
#             cycle_zebra_loss = l1(zebra, cycle_zebra)
#             cycle_horse_loss = l1(horse, cycle_horse)

#             # identity loss (remove these for efficiency if you set lambda_identity=0)
#             identity_zebra = gen_Z(zebra)
#             identity_horse = gen_H(horse)
#             identity_zebra_loss = l1(zebra, identity_zebra)
#             identity_horse_loss = l1(horse, identity_horse)

#             # add all togethor
#             G_loss = (
#                 loss_G_Z
#                 + loss_G_H
#                 + cycle_zebra_loss * config.LAMBDA_CYCLE
#                 + cycle_horse_loss * config.LAMBDA_CYCLE
#                 + identity_horse_loss * config.LAMBDA_IDENTITY
#                 + identity_zebra_loss * config.LAMBDA_IDENTITY
#             )

#         opt_gen.zero_grad()
#         g_scaler.scale(G_loss).backward()
#         g_scaler.step(opt_gen)
#         g_scaler.update()

#         if idx % 200 == 0:
#             save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
#             save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")

#         loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


# def main():
#     disc_H = Discriminator(in_channels=3).to(config.DEVICE)
#     disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
#     gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
#     gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
#     opt_disc = optim.Adam(
#         list(disc_H.parameters()) + list(disc_Z.parameters()),
#         lr=config.LEARNING_RATE,
#         betas=(0.5, 0.999),
#     )

#     opt_gen = optim.Adam(
#         list(gen_Z.parameters()) + list(gen_H.parameters()),
#         lr=config.LEARNING_RATE,
#         betas=(0.5, 0.999),
#     )

#     L1 = nn.L1Loss()
#     mse = nn.MSELoss()

#     if config.LOAD_MODEL:
#         load_checkpoint(
#             config.CHECKPOINT_GEN_H,
#             gen_H,
#             opt_gen,
#             config.LEARNING_RATE,
#         )
#         load_checkpoint(
#             config.CHECKPOINT_GEN_Z,
#             gen_Z,
#             opt_gen,
#             config.LEARNING_RATE,
#         )
#         load_checkpoint(
#             config.CHECKPOINT_CRITIC_H,
#             disc_H,
#             opt_disc,
#             config.LEARNING_RATE,
#         )
#         load_checkpoint(
#             config.CHECKPOINT_CRITIC_Z,
#             disc_Z,
#             opt_disc,
#             config.LEARNING_RATE,
#         )

#     dataset = HorseZebraDataset(
#         root_horse=config.TRAIN_DIR + "/horses",
#         root_zebra=config.TRAIN_DIR + "/zebras",
#         transform=config.transforms,
#     )
#     val_dataset = HorseZebraDataset(
#         root_horse=config.VAL_DIR + "/horses",
#         root_zebra=config.VAL_DIR + "/zebras",
#         transform=config.transforms,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=1,
#         shuffle=False,
#         pin_memory=True,
#     )
#     loader = DataLoader(
#         dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=True,
#     )
#     g_scaler = torch.cuda.amp.GradScaler()
#     d_scaler = torch.cuda.amp.GradScaler()

#     for epoch in range(config.NUM_EPOCHS):
#         train_fn(
#             disc_H,
#             disc_Z,
#             gen_Z,
#             gen_H,
#             loader,
#             opt_disc,
#             opt_gen,
#             L1,
#             mse,
#             d_scaler,
#             g_scaler,
#         )

#         if config.SAVE_MODEL:
#             save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
#             save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
#             save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
#             save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


# if __name__ == "__main__":
#     main()


import torch
<<<<<<< Updated upstream
from dataset import Pix2Pix
=======
from dataset import ObjectToObjectDataset
>>>>>>> Stashed changes
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


<<<<<<< Updated upstream
def train_fn(
        disc_obj1, disc_obj2, gen_obj1, gen_obj2, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, object_1, object_2
):
=======
def train_fn(disc_obj1, disc_obj2, gen_obj1, gen_obj2, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, object_1,
             object_2):
>>>>>>> Stashed changes
    obj1_reals = 0
    obj1_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (obj1, obj2) in enumerate(loop):
        obj1 = obj1.to(config.DEVICE)
        obj2 = obj2.to(config.DEVICE)

        # Train Discriminators
<<<<<<< Updated upstream
        with torch.cuda.amp.autocast():
=======
        with torch.cuda.amp.autocast():  # torch.cuda.amp.autocast is for float 16
>>>>>>> Stashed changes
            fake_obj1 = gen_obj1(obj2)
            D_obj1_real = disc_obj1(obj1)
            D_obj1_fake = disc_obj1(fake_obj1.detach())
            obj1_reals += D_obj1_real.mean().item()
            obj1_fakes += D_obj1_fake.mean().item()
            D_obj1_real_loss = mse(D_obj1_real, torch.ones_like(D_obj1_real))
            D_obj1_fake_loss = mse(D_obj1_fake, torch.zeros_like(D_obj1_fake))
            D_obj1_loss = D_obj1_real_loss + D_obj1_fake_loss

<<<<<<< Updated upstream





            fake_obj2 = gen_obj2(obj1)
            D_obj2_real = disc_obj2(obj2)
            D_obj2_fake = disc_obj2(fake_obj2.detach())
            D_obj2_real_loss = mse(D_obj2_real, torch.ones_like(D_obj2_real))
            D_obj2_fake_loss = mse(D_obj2_fake, torch.zeros_like(D_obj2_fake))
            D_obj2_loss = D_obj2_real_loss + D_obj2_fake_loss

            # Combine losses
            D_loss = (D_obj1_loss + D_obj2_loss) / 2
=======
            fake_obj2 = gen_obj2(obj1)
            D_obj2_real = disc_obj2(obj2)
            D_obj2_fake = disc_obj2(fake_obj2.detach())
            D_obj2_real_loss = mse(D_obj2_real, torch.ones_like(D_obj2_real))
            D_obj2_fake_loss = mse(D_obj2_fake, torch.zeros_like(D_obj2_fake))
            D_obj2_loss = D_obj2_real_loss + D_obj2_fake_loss

            # Combined loss
            D_loss = (D_obj1_loss + D_obj2_loss) / 2  # divided by 2 mentioned in the paper
>>>>>>> Stashed changes

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_obj1_fake = disc_obj1(fake_obj1)
            D_obj2_fake = disc_obj2(fake_obj2)
            loss_G_obj1 = mse(D_obj1_fake, torch.ones_like(D_obj1_fake))
            loss_G_obj2 = mse(D_obj2_fake, torch.ones_like(D_obj2_fake))

<<<<<<< Updated upstream

            # Cycle loss
            cycle_obj1 = gen_obj1(fake_obj2)
            cycle_obj2 = gen_obj2(fake_obj1)
            cycle_obj1_loss = l1(obj1, cycle_obj1)
            cycle_obj2_loss = l1(obj2, cycle_obj2)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_obj1 = gen_obj1(obj1)
            identity_obj2 = gen_obj2(obj2)
            identity_obj1_loss = l1(obj1, identity_obj1)
            identity_obj2_loss = l1(obj2, identity_obj2)

=======
            # Cycle loss
            cycle_obj1 = gen_obj1(fake_obj2)
            cycle_obj2 = gen_obj2(fake_obj1)
            cycle_obj1_loss = l1(obj1, cycle_obj1)
            cycle_obj2_loss = l1(obj2, cycle_obj2)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_obj1 = gen_obj1(obj1)
            identity_obj2 = gen_obj2(obj2)
            identity_obj1_loss = l1(obj1, identity_obj1)
            identity_obj2_loss = l1(obj2, identity_obj2)

>>>>>>> Stashed changes
            # Combine all losses
            G_loss = (
                    loss_G_obj1
                    + loss_G_obj2
                    + cycle_obj1_loss * config.LAMBDA_CYCLE
                    + cycle_obj2_loss * config.LAMBDA_CYCLE
<<<<<<< Updated upstream
                    + identity_obj1_loss
                    + identity_obj2_loss)

=======
                    + identity_obj1_loss * config.LAMBDA_IDENTITY
                    + identity_obj2_loss * config.LAMBDA_IDENTITY)
>>>>>>> Stashed changes

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
<<<<<<< Updated upstream
            save_image(fake_obj1 * 0.5 + 0.5, f"saved_images/{object_1}_{idx}.png")
            save_image(fake_obj2 * 0.5 + 0.5, f"saved_images/{object_2}_{idx}.png")
=======
            save_image(fake_obj1 * 0.5 + 0.5, f"saved_images/fake_{object_1}_{idx}.png")
            save_image(fake_obj2 * 0.5 + 0.5, f"saved_images/fake_{object_2}_{idx}.png")
>>>>>>> Stashed changes

        loop.set_postfix(obj1_real=obj1_reals / (idx + 1), obj1_fake=obj1_fakes / (idx + 1))


<<<<<<< Updated upstream




def main(object_1:str, object_2:str):

    disc_obj1 = Discriminator(in_channels=3).to(config.DEVICE)
    disc_obj2 = Discriminator(in_channels=3).to(config.DEVICE)
    gen_obj1 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_obj2 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
=======
def main(object_1: str, object_2: str):
    disc_obj1 = Discriminator(in_channels=3).to(config.DEVICE)  # discriminator check if there is real or fake obj1
    disc_obj2 = Discriminator(in_channels=3).to(config.DEVICE)  # discriminator check if there is real or fake obj2
    gen_obj1 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # generator generates a fake obj1
    gen_obj2 = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)  # generator generates a fake obj2
>>>>>>> Stashed changes
    opt_disc = optim.Adam(
        list(disc_obj1.parameters()) + list(disc_obj2.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),  # these values specified in paper
    )

    opt_gen = optim.Adam(
        list(gen_obj1.parameters()) + list(gen_obj2.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_OBJ1,
            gen_obj1,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_OBJ2,
            gen_obj2,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_OBJ1,
            disc_obj1,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_OBJ2,
            disc_obj2,
            opt_disc,
            config.LEARNING_RATE,
        )

<<<<<<< Updated upstream
    dataset = Pix2Pix(
=======
    dataset = ObjectToObjectDataset(
>>>>>>> Stashed changes
        root_obj1=config.TRAIN_DIR + '/' + object_1,
        root_obj2=config.TRAIN_DIR + '/' + object_2,
        transform=config.transforms,
    )
<<<<<<< Updated upstream
    val_dataset = Pix2Pix(
        root_obj1=config.VAL_DIR + '/' + object_1,
        root_obj2=config.VAL_DIR + '/' + object_2,
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
=======
    # val_dataset = ObjectToObjectDataset(
    #     root_obj1=config.VAL_DIR + '/' + object_1,
    #     root_obj2=config.VAL_DIR + '/' + object_2,
    #     transform=config.transforms,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
>>>>>>> Stashed changes
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print('epoch: ', epoch)
        train_fn(
            disc_obj1,
            disc_obj2,
            gen_obj1,
            gen_obj2,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            object_1,
<<<<<<< Updated upstream
            object_2
=======
            object_2,
>>>>>>> Stashed changes
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_obj1, opt_gen, filename=config.CHECKPOINT_GEN_OBJ1)
            save_checkpoint(gen_obj2, opt_gen, filename=config.CHECKPOINT_GEN_OBJ2)
            save_checkpoint(disc_obj1, opt_disc, filename=config.CHECKPOINT_DISC_OBJ1)
            save_checkpoint(disc_obj2, opt_disc, filename=config.CHECKPOINT_DISC_OBJ2)


if __name__ == "__main__":
<<<<<<< Updated upstream
    main('happy', 'sad')
=======
    main('horses', 'zebras')
>>>>>>> Stashed changes
