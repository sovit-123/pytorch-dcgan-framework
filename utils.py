import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

plt.style.use('ggplot')

def label_real(size):
    """
    Fucntion to create real labels (ones)
    :param size: batch size
    :return real label vector
    """
    data = torch.ones(size)
    return data

def label_fake(size):
    """
    Fucntion to create fake labels (zeros)
    :param size: batch size
    :returns fake label vector
    """
    data = torch.zeros(size)
    return data

# function to create the noise vector
def create_noise(sample_size, nz):
    """
    Fucntion to create noise
    :param sample_size: fixed sample size or batch size
    :param nz: latent vector size
    :returns random noise vector
    """
    return torch.randn(sample_size, nz, 1, 1)

def save_generator_image(image, path):
    """
    Function to save torch image batches
    :param image: image tensor batch
    :param path: path name to save image
    """
    save_image(
        image, path, 
        normalize=True
    )

def make_output_dir(dir_name=None):
    """
    Function to create the output directory to store the
    generated image, the final GIF, the generator model, and the 
    loss graph

    Parameters:
    :param dir_name: path to the directory
    """
    os.makedirs(f"outputs_{dir_name}", exist_ok=True)


def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. This follows the specification from the DCGAN paper.
    https://arxiv.org/pdf/1511.06434.pdf
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def print_params(model, model_name=None):
    """
    Function to print the total number of parameters and trainable 
    parameters in a model.

    Parameters
    :param model: PyTorch model instance.
    :parm model_name: Name of the model. A string.
    """
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MODEL: {model_name}")
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

def save_loss_plots(gen_loss, disc_loss, path):
    """
    Function to save the plots to disk.

    Parameters
    :param plot_lost: List containing the values.
    :param name: Path to save the plot.
    """
    # plot and save the generator and discriminator loss
    plt.figure(figsize=(10, 7))
    plt.plot(gen_loss, label='Generator loss')
    plt.plot(disc_loss, label='Discriminator Loss')
    plt.legend()
    plt.savefig(path)

def initialize_tensorboard(DATASET):
    """
    Function to initialize the TensorBoard SummaryWriter.

    Parameters
    :param DATASET: String dataset name to act as subpath.
    :return the SummaryWriter object.
    """
    os.makedirs(f"runs/{DATASET}", exist_ok=True)
    num_dirs = os.listdir(f"runs/{DATASET}")
    writer = SummaryWriter(f"runs/{DATASET}/run_{len(num_dirs)+1}")
    return writer

def add_tensorboard_scalar(loss_name, writer, loss, n_step):
    writer.add_scalars(loss_name, loss, n_step)

def save_gen_model(epochs, model, optimizer, criterion, path):
    # save model checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_fn': criterion,
    }, path)

def set_resume_training(model_path):
    checkpoint = torch.load(model_path)
    epochs_trained = checkpoint['epoch']
    return epochs_trained
