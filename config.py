import torch

BATCH_SIZE = 128
EPOCHS  = 50
EPOCH_START = 0
NUM_WORKERS = 4
MULT_FACTOR = 1
IMAGE_SIZE = 64*MULT_FACTOR
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
# one of 'MNIST', 'FashionMNIST', 'CIFAR10', 'CELEBA', ...
# ... 'ABSTRACT_ART'
DATASET = 'CELEBA'

# for printing metrics
PRINT_EVERY = 100

# for optimizer
BETA1 = 0.5
BETA2 = 0.999
LEARNING_RATE = 0.0002

# Epcoh nterval at which to save the Generator Model.
MODEL_SAVE_INTERVAL = 25

# Provide path to a trained model to resume training, else keep `None`.
# GEN_MODEL_PATH = 'outputs_MNIST/generator_final.pth'
# DISC_MODEL_PATH = 'outputs_MNIST/discriminator_final.pth'
GEN_MODEL_PATH = None
DISC_MODEL_PATH = None

# Whether to create GIF from all the generated images at the end or not,
# might need a considerable amoung of RAM as all the generated images will
# be loaded to at once. Give values as either `True` or `False`.
CREATE_GIF = False