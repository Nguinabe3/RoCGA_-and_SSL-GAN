#import necessary librairies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Define transformations for the training set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Define the number of classes
num_classes = 10

class Encoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, latent_dim)
        )

    def forward(self, img):
        z = self.model(img)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Decoder, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Model parameters
latent_dim = 100
img_shape = (1, 28, 28)

# Initialize models
encoder = Encoder(latent_dim, img_shape)
decoder = Decoder(latent_dim, img_shape)
generator = Generator(latent_dim, num_classes, img_shape)
discriminator = Discriminator(num_classes, img_shape)

# Random initialization (placeholder for pretrained weights)
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Initialize all networks
encoder.apply(weights_init_normal)
decoder.apply(weights_init_normal)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss functions
adversarial_loss = nn.MSELoss()
reconstruction_loss = nn.L1Loss()

# Optimizers
optimizer_E = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Dec = optim.Adam(decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)
reconstruction_loss.to(device)

# Training parameters
n_epochs = 200
sample_interval = 400

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(trainloader):
        
        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1, device=device, requires_grad=False)
        fake = torch.zeros(imgs.size(0), 1, device=device, requires_grad=False)
        
        # Configure input
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        
        # ---------------------
        #  Train Encoder and Generator
        # ---------------------
        
        optimizer_E.zero_grad()
        optimizer_Dec.zero_grad()
        optimizer_G.zero_grad()
        
        # Sample noise and labels as generator input
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (imgs.size(0),), device=device)
        
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        
        # Encode real images
        encoded_imgs = encoder(real_imgs)
        recon_imgs = decoder(encoded_imgs)
        
        # Loss measures generator's ability to fool the discriminator and reconstruction loss
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid) + reconstruction_loss(recon_imgs, real_imgs)
        
        g_loss.backward()
        optimizer_G.step()
        optimizer_E.step()
        optimizer_Dec.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        optimizer_D.zero_grad()
        
        # Loss for real images
        real_pred = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_pred, valid)
        
        # Loss for fake images
        fake_pred = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_pred, fake)
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        # Print the progress
        if i % sample_interval == 0:
            print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(trainloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")


# Function to generate and save images
def sample_image(num, n_row=10, epoch=n_epochs):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    z = torch.randn(n_row ** 2, latent_dim, device=device)
    labels = torch.arange(0, num_classes).repeat(n_row).to(device)
    gen_imgs = generator(z, labels)
    gen_imgs = gen_imgs.view(gen_imgs.size(0), *img_shape)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images 0 - 1
    plt.imshow(gen_imgs[num].cpu().detach().squeeze(), cmap='gray')
    plt.axis(False)
    plt.show()
sample_image(5, n_row=10, epoch=n_epochs)