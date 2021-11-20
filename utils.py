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

    :param size: Batch size.

    Returns:
        Real label vector
    """
    data = torch.ones(size)
    return data

def label_fake(size):
    """
    Fucntion to create fake labels (zeros)

    :param size: Batch size.

    Returns:
        Fake label vector
    """
    data = torch.zeros(size)
    return data

# function to create the noise vector
def create_noise(sample_size, nz):
    """
    Fucntion to create noise

    :param sample_size: Fixed sample size or batch size.
    :param nz: Latent vector size.

    Returns:
        Random noise vector
    """
    return torch.randn(sample_size, nz, 1, 1)

def save_generator_image(image, path):
    """
    Function to save torch image batches

    :param image: Image tensor batch.
    :param path: Path name to save image.
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

    :param dir_name: Path to the directory.
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

    :param DATASET: String dataset name to act as subpath.
    Returns:
        The SummaryWriter object.
    """
    os.makedirs(f"runs/{DATASET}", exist_ok=True)
    num_dirs = os.listdir(f"runs/{DATASET}")
    writer = SummaryWriter(f"runs/{DATASET}/run_{len(num_dirs)+1}")
    return writer

def add_tensorboard_scalar(loss_name, writer, loss, n_step):
    writer.add_scalars(loss_name, loss, n_step)

def save_model(
    epochs, model, optimizer, criterion, 
    losses, batch_losses, path
):
    """
    Save the model to disk along with other properties..

    :param epochs: Number of epochs trained for.
    :param model: The neural network model.
    :param optimizer: The optimizer instance.
    :param criterion: The loss function instance.
    :param losses: List containing loss values for each epoch.
    :param batch_losses: List containing batch-wise loss values.
    :param path: String. Path to save the model
    """
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'batch_losses': batch_losses,
        'loss_fn': criterion,
    }, path)

def set_resume_training(model_path):
    """
    This function is executed if a trained model path is provided to
    `MODEL_PATH` in `config.py`. This sets up every required variable to
    resume training.

    :param gen_model_path: Path to the trained generator model

    Returns:
        epochs_trained: Number of epochs already trained for.
        model_state_dict: Trained state dictionary of the model.
        optimizer_state_dict: Trained optimizer state.
        losses: List containing epoch-wise losses.
        batch_losses: List containing batch-wise losses.
    """
    checkpoint = torch.load(model_path)
    epochs_trained = checkpoint['epoch'] # Number of epochs trained for.
    model_state_dict = checkpoint['model_state_dict'] # Trained weights.
    optimizer_state_dict = checkpoint['optimizer_state_dict'] # Optimizer state.
    losses = checkpoint['losses'] # Available epoch-wise losses, a list.
    batch_losses = checkpoint['batch_losses'] # Available batch-wise losses.
    return (
        epochs_trained, model_state_dict, optimizer_state_dict, 
        losses, batch_losses
    )