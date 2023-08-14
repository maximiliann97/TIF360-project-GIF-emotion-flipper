import torch
import torch.nn as nn
import configurations
from tqdm import tqdm
from torchvision.utils import save_image


class Trainer:
    def __init__(self, disc_obj1, disc_obj2, gen_obj1, gen_obj2, loader, opt_disc, opt_gen, d_scaler,
                 g_scaler, l1, mse):
        self.disc_obj1 = disc_obj1
        self.disc_obj2 = disc_obj2
        self.gen_obj1 = gen_obj1
        self.gen_obj2 = gen_obj2
        self.loader = loader
        self.opt_disc = opt_disc
        self.opt_gen = opt_gen
        self.d_scaler = d_scaler
        self.g_scaler = g_scaler
        self.l1 = l1
        self.mse = mse

    def train(self, object_1, object_2, num_epochs):
        for epoch in range(num_epochs):
            obj1_reals = 0
            obj1_fakes = 0
            loop = tqdm(self.loader, leave=True)

            for idx, (obj1, obj2) in enumerate(loop):
                obj1 = obj1.to(configurations.DEVICE)
                obj2 = obj2.to(configurations.DEVICE)

                # Train Discriminators
                with torch.cuda.amp.autocast():
                    fake_obj1 = self.gen_obj1(obj2)
                    D_obj1_real = self.disc_obj1(obj1)
                    D_obj1_fake = self.disc_obj1(fake_obj1.detach())
                    obj1_reals += D_obj1_real.mean().item()
                    obj1_fakes += D_obj1_fake.mean().item()
                    D_obj1_real_loss = self.mse(D_obj1_real, torch.ones_like(D_obj1_real))
                    D_obj1_fake_loss = self.mse(D_obj1_fake, torch.zeros_like(D_obj1_fake))
                    D_obj1_loss = D_obj1_real_loss + D_obj1_fake_loss

                    fake_obj2 = self.gen_obj2(obj1)
                    D_obj2_real = self.disc_obj2(obj2)
                    D_obj2_fake = self.disc_obj2(fake_obj2.detach())
                    D_obj2_real_loss = self.mse(D_obj2_real, torch.ones_like(D_obj2_real))
                    D_obj2_fake_loss = self.mse(D_obj2_fake, torch.zeros_like(D_obj2_fake))
                    D_obj2_loss = D_obj2_real_loss + D_obj2_fake_loss

                    # Combine losses
                    D_loss = (D_obj1_loss + D_obj2_loss) / 2

                self.opt_disc.zero_grad()
                self.d_scaler.scale(D_loss).backward()
                self.d_scaler.step(self.opt_disc)
                self.d_scaler.update()

                # Train Generators
                with torch.cuda.amp.autocast():
                    # Adversarial loss for both generators
                    D_obj1_fake = self.disc_obj1(fake_obj1)
                    D_obj2_fake = self.disc_obj2(fake_obj2)
                    loss_G_obj1 = self.mse(D_obj1_fake, torch.ones_like(D_obj1_fake))
                    loss_G_obj2 = self.mse(D_obj2_fake, torch.ones_like(D_obj2_fake))

                    # Cycle loss
                    cycle_obj1 = self.gen_obj1(fake_obj2)
                    cycle_obj2 = self.gen_obj2(fake_obj1)
                    cycle_obj1_loss = self.l1(obj1, cycle_obj1)
                    cycle_obj2_loss = self.l1(obj2, cycle_obj2)

                    # Identity loss (remove these for efficiency if you set lambda_identity=0)
                    # identity_obj1 = gen_obj1(obj1)
                    # identity_obj2 = gen_obj2(obj2)
                    # identity_obj1_loss = l1(obj1, identity_obj1)
                    # identity_obj2_loss = l1(obj2, identity_obj2)

                    # Combine all losses
                    G_loss = (
                            loss_G_obj1
                            + loss_G_obj2
                            + cycle_obj1_loss * configurations.LAMBDA_CYCLE
                            + cycle_obj2_loss * configurations.LAMBDA_CYCLE
                        # + identity_obj1_loss
                        # + identity_obj2_loss
                    )

                self.opt_gen.zero_grad()
                self.g_scaler.scale(G_loss).backward()
                self.g_scaler.step(self.opt_gen)
                self.g_scaler.update()

                if idx % 300 == 0:
                    save_image(fake_obj1 * 0.5 + 0.5, f"/content/drive/MyDrive/saved_images/{object_1}_{idx}.png")
                    save_image(fake_obj2 * 0.5 + 0.5, f"/content/drive/MyDrive/saved_images/{object_2}_{idx}.png")

                loop.set_postfix(obj1_real=obj1_reals / (idx + 1), obj1_fake=obj1_fakes / (idx + 1))

            # Save checkpoints after each epoch
            if epoch % 50 == 0:
                self.save_checkpoint(self.gen_obj1, self.opt_gen, f"/content/drive/MyDrive/generators/epoch_{epoch}_gen_{object_1}.pth.tar")
                self.save_checkpoint(self.gen_obj2, self.opt_gen, f"/content/drive/MyDrive/generators/epoch_{epoch}_gen_{object_2}.pth.tar")
                self.save_checkpoint(self.disc_obj1, self.opt_disc, f"/content/drive/MyDrive/discriminators/epoch_{epoch}_disc_{object_1}.pth.tar")
                self.save_checkpoint(self.disc_obj2, self.opt_disc, f"/content/drive/MyDrive/discriminators/epoch_{epoch}_disc_{object_2}.pth.tar")
                self.save_model(self.gen_obj1, f"/content/drive/MyDrive/generator_models/epoch_{epoch}_gen_{object_1}.pth.tar")
                self.save_model(self.gen_obj2, f"/content/drive/MyDrive/generator_models/epoch_{epoch}_gen_{object_2}.pth.tar")

            self.save_checkpoint(self.gen_obj1, self.opt_gen, f"/content/drive/MyDrive/generators/gen_{object_1}.pth.tar")
            self.save_checkpoint(self.gen_obj2, self.opt_gen, f"/content/drive/MyDrive/generators/gen_{object_2}.pth.tar")
            self.save_checkpoint(self.disc_obj1, self.opt_disc, f"/content/drive/MyDrive/discriminators/disc_{object_1}.pth.tar")
            self.save_checkpoint(self.disc_obj2, self.opt_disc, f"/content/drive/MyDrive/discriminators/disc_{object_2}.pth.tar")
            self.save_model(self.gen_obj1, f"/content/drive/MyDrive/generator_models/gen_{object_1}.pth.tar")
            self.save_model(self.gen_obj2, f"/content/drive/MyDrive/generator_models/gen_{object_2}.pth.tar")

    def save_checkpoint(self, model, optimizer, filename):
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, model, optimizer, filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    def save_model(self, model, filename):
        torch.save(model, filename)


