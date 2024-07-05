import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 3 * 64 * 64),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 64 * 64, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.adversarial_loss = nn.BCELoss()
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for i, imgs in enumerate(dataloader):
                valid = torch.ones((imgs.size(0), 1))
                fake = torch.zeros((imgs.size(0), 1))

                # Train Generator
                self.g_optimizer.zero_grad()
                z = torch.randn(imgs.size(0), self.latent_dim)
                gen_imgs = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.g_optimizer.step()

                # Train Discriminator
                self.d_optimizer.zero_grad()
                real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()

            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def generate(self, audio_features, emotion):
        z = self.audio_to_gan_input(audio_features, emotion)
        with torch.no_grad():
            gen_img = self.generator(z)
        return gen_img.squeeze().permute(1, 2, 0).numpy()

    def audio_to_gan_input(self, audio_features, emotion):
        gan_input = np.zeros(self.latent_dim)
        gan_input[:33] = audio_features
        gan_input[33:37] = emotion
        gan_input[37:] = np.random.normal(0, 1, self.latent_dim - 37)
        return torch.FloatTensor(gan_input).unsqueeze(0)

# Note: You'll need to train this GAN with a dataset of abstract art before using it