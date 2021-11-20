from models import Generator, Discriminator
from config import(
    IMAGE_SIZE, NZ, DEVICE, SAMPLE_SIZE, 
    EPOCHS, K, BATCH_SIZE, DATASET, 
    NUM_WORKERS, PRINT_EVERY, BETA1, BETA2,
    N_CHANNELS, LEARNING_RATE, MODEL_SAVE_INTERVAL,
    EPOCH_START, GEN_MODEL_PATH, DISC_MODEL_PATH, 
    CREATE_GIF
)
from utils import (
    label_fake, label_real, create_noise,
    save_generator_image, make_output_dir, 
    weights_init, print_params, save_loss_plots,
    initialize_tensorboard, add_tensorboard_scalar,
    save_model, set_resume_training
)
from datasets import return_data

from torchvision.utils import make_grid
from PIL import Image

import torch.optim as optim
import torch.nn as nn
import imageio
import glob as glob
import time
import random
import torch
import numpy as np

# function to train the discriminator network
def train_discriminator(
    optimizer, data_real, data_fake, label_fake, label_real
):
    # get the batch size
    b_size = data_real.size(0)
    # get the real label vector
    real_label = label_real(b_size).to(DEVICE)
    # print(real_label.shape)
    # get the fake label vector
    fake_label = label_fake(b_size).to(DEVICE)

    optimizer.zero_grad()

    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real).view(-1)
    loss_real = criterion(output_real, real_label)
    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake).view(-1)
    loss_fake = criterion(output_fake, fake_label)

    # real loss backprop
    loss_real.backward()
    # fake data loss backprop
    loss_fake.backward()
    # update discriminator parameters
    optimizer.step()

    loss = loss_real + loss_fake
    return loss

# function to train the generator network
def train_generator(optimizer, data_fake, label_real):
    # get the batch size
    b_size = data_fake.size(0)
    # get the real label vector
    real_label = label_real(b_size).to(DEVICE)

    optimizer.zero_grad()

    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake).view(-1)
    loss = criterion(output, real_label)

    # backprop 
    loss.backward()
    # update generator parameters
    optimizer.step()

    return loss

if __name__ == '__main__':
    # Set seed
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # initialize the generator
    generator = Generator(
        NZ, IMAGE_SIZE, N_CHANNELS
    ).to(DEVICE)
    # initialize the discriminator
    discriminator = Discriminator(N_CHANNELS).to(DEVICE)
    # initialize generator weights
    generator.apply(weights_init)
    # initialize discriminator weights
    discriminator.apply(weights_init)

        # optimizers
    optim_g = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)
    )
    optim_d = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)
    )

    # loss function
    criterion = nn.BCELoss() 

    losses_g = [] # to store generator loss after each epoch
    losses_d = [] # to store discriminator loss after each epoch
    batch_losses_g = [] # to store generator loss after each batch
    batch_losses_d = [] # to store discriminator loss after each batch
    images = [] # to store images generatd by the generator   

    global_batch_iter = 0

    # Initialize SummaryWriter.
    writer = initialize_tensorboard(DATASET)

    # If path is provided, then training will resume from that
    # provided model's state dictionary.
    if GEN_MODEL_PATH:
        print('[NOTE]: Resuming training...\n')
        (
            epochs_trained, gen_state_dict, 
            gen_optim_state_dict, losses_g, batch_losses_g,
            global_batch_iter
        ) = set_resume_training(GEN_MODEL_PATH)
        if EPOCHS <= epochs_trained:
            print(f"Models already trained for {epochs_trained} epochs...")
            print(f"Enter more number of epochs than {epochs_trained}")
        error_string = f"Models already trained for {epochs_trained} epochs..."
        error_string += f" Enter more number of epochs than {epochs_trained}"
        assert EPOCHS > epochs_trained, error_string
        
        (
            _, disc_state_dict,
            disc_optim_state_dict, losses_d, batch_losses_d,
            global_batch_iter
        ) = set_resume_training(DISC_MODEL_PATH)
        EPOCH_START = epochs_trained

        # Load the trained weights into the models.
        generator.load_state_dict(gen_state_dict)
        discriminator.load_state_dict(disc_state_dict)
        # Load the trained optimizer states.
        optim_g.load_state_dict(gen_optim_state_dict)
        optim_d.load_state_dict(disc_optim_state_dict)

        # Add the previous TensorBoard logs to current one,
        # for continuity.
        print('[NOTE]: Adding previous TensorBoard logs to current session...')
        for batch_iter in range(global_batch_iter):
            add_tensorboard_scalar(
                    'Batch_Loss', writer, 
                    {'gen_batch_loss': np.array(batch_losses_g[batch_iter]), 
                    'disc_batch_loss': np.array(batch_losses_d[batch_iter])}, 
                    batch_iter
                )
        for epoch_iter in range(epochs_trained):
            add_tensorboard_scalar(
                'Epoch_Loss', writer, 
                {'gen_epoch_loss': np.array(losses_g[epoch_iter]), 
                'disc_epoch_loss': np.array(losses_d[epoch_iter])}, 
                epoch_iter 
            )

    print('##### GENERATOR #####')
    print(generator)
    print_params(generator, 'Generator')
    print('######################')

    print('\n##### DISCRIMINATOR #####')
    print(discriminator)
    print_params(discriminator, 'Discriminator')
    print('######################')

    generator.train()
    discriminator.train()

    # create the noise vector
    noise = create_noise(SAMPLE_SIZE, NZ).to(DEVICE)

    # train data loader
    train_loader = return_data(
        BATCH_SIZE, data=DATASET, 
        num_worders=NUM_WORKERS, image_size=IMAGE_SIZE
    )

    # create directory to save generated images, trained generator model...
    # ... and the loss graph
    make_output_dir(DATASET)

    for epoch in range(EPOCH_START, EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")
        epoch_start = time.time()
        loss_g = 0.0
        loss_d = 0.0
        for bi, data in enumerate(train_loader):
            print(f"Batches: [{bi+1}/{len(train_loader)}]", end='\r')
            image, _ = data
            image = image.to(DEVICE)
            b_size = len(image)
            # run the discriminator for k number of steps
            for step in range(K):
                data_fake = generator(create_noise(b_size, NZ).to(DEVICE)).detach()
                data_real = image
                # train the discriminator network
                bi_loss_d = train_discriminator(
                    optim_d, data_real, data_fake, label_fake, label_real
                )

                # add current discriminator batch loss to `loss_d`
                loss_d += bi_loss_d
                # append current discriminator batch loss to `batch_losses_d`
                batch_losses_d.append(bi_loss_d.detach().cpu())

            data_fake = generator(create_noise(b_size, NZ).to(DEVICE))
            # train the generator network
            bi_loss_g = train_generator(optim_g, data_fake, label_real)
            # add current generator batch loss to `loss_g`
            loss_g += bi_loss_g
            # append current generator batch loss to `batch_losses_g`
            batch_losses_g.append(bi_loss_g.detach().cpu())

            # Add each batch Generator and Discriminator loss to TensorBoard
            add_tensorboard_scalar(
                'Batch_Loss', writer, 
                {'gen_batch_loss': bi_loss_g, 'disc_batch_loss': bi_loss_d}, 
                global_batch_iter
            )
            
            # Print the loss of the current iteration after `PRINT_EVERY` iterations.
            if (bi+1) % PRINT_EVERY == 0:
                print(f"[Epoch/Epochs] [{epoch+1}/{EPOCHS}], [Batch/Batches] [{bi+1}/{len(train_loader)}], Gen_loss: {bi_loss_g}, Disc_loss: {bi_loss_d}")
            global_batch_iter += 1

        # Save the models after the specified interval.
        if (epoch+1) % MODEL_SAVE_INTERVAL == 0:
            save_model(
                epoch+1, generator, optim_g, criterion, 
                losses_g, batch_losses_g, global_batch_iter,
                f"outputs_{DATASET}/generator_{epoch+1}.pth"
            )
            save_model(
                epoch+1, discriminator, optim_d, criterion, 
                losses_d, batch_losses_d, global_batch_iter,
                f"outputs_{DATASET}/discriminator_{epoch+1}.pth"
            )

        # create the final fake image for the epoch
        generated_img = generator(noise).cpu().detach()
        # make the images as grid
        generated_img = make_grid(generated_img)
        # save the generated torch tensor models to disk
        save_generator_image(generated_img, f"outputs_{DATASET}/gen_img{epoch+1}.png")
        
        epoch_loss_g = loss_g / bi # total generator loss for the epoch
        epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
        # Append current generator epoch loss to list.
        losses_g.append(epoch_loss_g.detach().cpu())
        # Append current discriminator epoch loss to list.
        losses_d.append(epoch_loss_d.detach().cpu())
        add_tensorboard_scalar(
            'Epoch_Loss', writer, 
            {'gen_epoch_loss': epoch_loss_g, 'disc_epoch_loss': epoch_loss_d}, 
            epoch
        )
        epoch_end = time.time()

        # Save the models final time.
        if (epoch+1) == EPOCHS:
            save_model(
                EPOCHS, generator, optim_g, criterion, 
                losses_g, batch_losses_g, global_batch_iter,
                f"outputs_{DATASET}/generator_final.pth"
            )
            save_model(
                EPOCHS, discriminator, optim_d, criterion, 
                losses_d, batch_losses_d, global_batch_iter,
                f"outputs_{DATASET}/discriminator_final.pth"
            )
        
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}\n")
        print(f"Took {(epoch_end-epoch_start):.3f} seconds for epoch {epoch+1}")
        print('-'*50, end='\n')
    print('DONE TRAINING')

    # save epoch loss plot
    save_loss_plots(losses_g, losses_d, f"outputs_{DATASET}/epoch_loss.png")
    # save batch loss plot
    save_loss_plots(
        batch_losses_g, batch_losses_d, f"outputs_{DATASET}/batch_loss.png"
    )

    # Save the generated images as GIF file if `CREATE_GIF` is `True`.
    if CREATE_GIF:
        all_saved_image_paths = glob.glob(f"outputs_{DATASET}/gen_*.png")
        imgs = [Image.open(image_path) for image_path in all_saved_image_paths]
        imageio.mimsave(f"outputs_{DATASET}/generator_images.gif", imgs)