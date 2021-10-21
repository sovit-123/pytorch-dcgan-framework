import torch

BATCH_SIZE = 128
EPOCHS  = 5
NUM_WORKERS = 4
IMAGE_SIZE = 64*1
# image channels
N_CHANNELS = 3

# SAMPLE_SIZE is the total number of images in row x column form...
# if SAMPLE_SIZE = 64, then 8x8 image grids will be saved to disk...
# if SAMPLE_SIZE = 128, then 16x8 image grids will be saved to disk...
SAMPLE_SIZE = 64
# latent vector size
NZ = 100
# number of steps to apply to the discriminator
K = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# one of 'MNIST', 'FashionMNIST', 'CIFAR10', 'CELEBA'
DATASET = 'CELEBA'

# for printing metrics
PRINT_EVERY = 100

# for optimizer
BETA1 = 0.5
BETA2 = 0.999
LEARNING_RATE = 0.0002