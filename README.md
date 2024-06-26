# RoCGAN and SLL-GAN Project Implementation (AMMI: 2023-2024)

## In this repository, we will discuss RoCGAN, SSL-GAN and its implementation.

### GAN

**Generative Adversarial Networks (GANs)** are a type of deep learning framework designed to generate realistic examples based on specific requirements. GANs consist of two main components that engage in a continuous, competitive process:

1. Generator Network: This neural network is responsible for creating new data. Depending on its training, it can generate various forms of data, such as images, text, videos, or sounds.

2. Discriminator Network: This neural network's task is to differentiate between real data from the dataset and the fake data produced by the generator.

### CGAN

**Conditional Generative Adversarial** : A conditional generative adversarial network (CGAN) is a type of GAN model where a condition is put in place to get the output.

### RoCGAN

The implementation of **Conditional Generative Adversarial Networks (CGANs)** often utilizes convolutional layers for image data or fully connected layers for other data types. These layers are designed to learn complex mappings from the source signal to the target signal. However, both convolutional and fully connected layers are sensitive to additive noise, meaning that even minor perturbations in the input signal can cause significant deviations in the output.

To address this, a condition, such as a label, is added to both the discriminator and the generator. This leads to the development of the **"Robust Conditional GAN" (RoCGAN)**, an enhanced version of the Conditional GAN (cGAN).

> Motivation for RoCGAN

The main motivation for developing RoCGAN is to tackle the sensitivity of cGANs to noise in the input signal. In real-world applications, input signals can often be noisy, resulting in reduced performance of the generated outputs. RoCGAN aims to enhance the model's robustness to such noise by integrating both supervised and unsupervised learning pathways.

> RoCGAN Architecture

RoCGAN enhances the traditional cGAN architecture by introducing a dual-pathway module that incorporates both supervised and unsupervised learning pathways. Here's a detailed breakdown of the architecture:

1. Supervised Pathway [Regression Pathway]:

 > Functions similarly to the generator in a standard cGAN, performing regression to map the source signal to the target signal.

> Includes an encoder that embeds the source signal into a latent representation. Features a decoder that maps this latent representation to the target space. 

2. Unsupervised Pathway [Autoencoder Pathway]:

> Acts as an autoencoder, learning the structure of the target domain in an unsupervised manner.
> Contains an encoder that processes samples from the target domain, embedding them into a latent space.
Has a decoder that reconstructs the original samples from these latent representations.

3. Shared Weights:

> The decoders in both pathways share weights, ensuring that the latent representations produced by both pathways are semantically similar.

> This shared structure helps constrain the output of the supervised pathway to lie within the target subspace, as learned through the unsupervised pathway.


### Semi-Supervised Learning with Generative Adversarial Networks

Semi-supervised learning is a machine learning approach where a model is trained using a combination of labeled and unlabeled data. This method aims to enhance the model's performance by utilizing both types of data, which is especially beneficial when labeled data is limited or costly to acquire.

**Generative Adversarial Networks (GANs)**, introduced by Ian Goodfellow and colleagues in 2014, are a neural network architecture consisting of two competing networks: the generator and the discriminator. The generator aims to produce realistic data samples, while the discriminator's task is to distinguish between real and generated data. Both networks are trained simultaneously in a competitive process, driving each other to improve.

When integrating semi-supervised learning with GANs, the goal is to utilize the GAN's generative capabilities to create additional labeled data. Here's a typical approach:

1. **Training the GAN:** Initially, the GAN is trained on a dataset containing only unlabeled data. The generator learns to produce data samples that mimic the distribution of the training data, while the discriminator learns to differentiate between real and generated data.

2. **Generating Synthetic Labeled Data:** After training, the generator can create synthetic data samples. These samples are labeled based on their classification by the discriminator. For instance, if the discriminator classifies a generated sample as belonging to a certain class, it is labeled accordingly.

3. **Combining Labeled and Unlabeled Data:** The synthetic labeled data generated by the GAN is combined with the original labeled data and the remaining unlabeled data. This results in a larger and more diverse dataset for training the model.

4. **Training the Classifier:** Finally, a classifier (often a neural network) is trained on this expanded dataset. The classifier learns to distinguish between different classes using the labeled data, the unlabeled data, and the synthetic labeled data generated by the GAN.

## Usage Case:

1. Clone the repository using: git clone https://github.com/Nguinabe3/RoCGA_-and_SSL-GAN.git

2. Go to clonned project using 'cd' command and run the following command:

>  python regular_gans_.py for the simple GAN model using MNIST dataset;

>  python RoCGAN_MNIST.py for the RoCGAN model using MNIST dataset;

>  python RoCGAN_CIFART.py for the RoCGAN model using CIFAR dataset;

>  python SSL_GANS.py for the SSL_GANS model using MNIST dataset;


# License: [MIT](https://choosealicense.com/licenses/mit/)
