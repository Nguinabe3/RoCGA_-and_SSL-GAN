#import necesary librairies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, np.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_shape, num_classes):
        super(Discriminator, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(np.prod(img_shape), 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.validity_predictor = nn.Linear(128, 1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        features = self.feature_extractor(img_flat)
        validity = self.validity_predictor(features)
        classes = self.classifier(features)
        return validity, classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
latent_dim = 100
img_shape = (1, 28, 28)  # Image shape for MNIST
n_epochs = 500
lr = 0.0002
batch_size = 64
num_classes = 10  # Number of classes for semi-supervised learning

# Load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape, num_classes).to(device)

# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()
classification_loss = nn.CrossEntropyLoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        # Move data to device
        imgs, labels = imgs.to(device), labels.to(device)

        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate noise
        z = torch.randn(imgs.size(0), latent_dim).to(device)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_labels = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_validity, real_pred_labels = discriminator(imgs)
        d_real_loss = adversarial_loss(real_validity, valid)
        real_classification_loss = classification_loss(real_pred_labels, labels)

        # Measure discriminator's ability to classify fake samples
        fake_validity, fake_pred_labels = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_validity, fake)

        # Total discriminator loss
        d_loss = 0.5 * (d_real_loss + d_fake_loss) + real_classification_loss

        d_loss.backward()
        optimizer_D.step()

        # Print progress
        if i % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )

    # Output generated images
    if epoch % 50 == 0:
        with torch.no_grad():
            z = torch.randn(10, latent_dim).to(device)
            gen_imgs = generator(z).cpu().numpy()

        fig, axs = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            axs[i].imshow(gen_imgs[i][0], cmap='gray')
            axs[i].axis('off')
        plt.show()
