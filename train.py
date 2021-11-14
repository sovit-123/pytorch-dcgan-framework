from models import Generator, Discriminator
from config import(
    IMAGE_SIZE, NZ, DEVICE, SAMPLE_SIZE, 
    EPOCHS, K, BATCH_SIZE, DATASET, 
    NUM_WORKERS, PRINT_EVERY, BETA1, BETA2,
    N_CHANNELS, LEARNING_RATE, GEN_MODEL_SAVE_INTERVAL
)
from utils import (
    label_fake, label_real, create_noise,
    save_generator_image, make_output_dir, 
    weights_init, print_params, save_loss_plots,
    initialize_tensorboard, add_tensorboard_scalar,
    save_gen_model
)
from datasets import return_data

from torchvision.utils import make_grid
from PIL import Image

import torch.optim as optim
import torch.nn as nn
import imageio
import torch
import glob as glob
import time

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
    # Initialize SummaryWriter.
    writer = initialize_tensorboard(DATASET)

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


    print('##### GENERATOR #####')
    print(generator)
    print_params(generator, 'Generator')
    print('######################')

    print('\n##### DISCRIMINATOR #####')
    print(discriminator)
    print_params(discriminator, 'Discriminator')
    print('######################')

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

    global_batch_iter = 0

    for epoch in range(EPOCHS):
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

            # Add average loss of Generator and Discriminator till current batch to TensorBoard
            # add_tensorboard_scalar(
            #     'Batch_Loss', writer, 
            #     {'gen_batch_loss': loss_g/(bi+1), 'disc_batch_loss': loss_d/(bi+1)}, 
            #     global_batch_iter
            # )

            # Add each batch Generator and Discriminator loss to TensorBoard
            add_tensorboard_scalar(
                'Batch_Loss', writer, 
                {'gen_batch_loss': bi_loss_g, 'disc_batch_loss': bi_loss_d}, 
                global_batch_iter
            )
            
            # Print average loss till `PRINT_EVERY` iterations.
            # if (bi+1) % PRINT_EVERY == 0:
            #     print(f"[Epoch/Epochs] [{epoch+1}/{EPOCHS}], [Batch/Batches] [{bi+1}/{len(train_loader)}], Gen_loss: {loss_g/bi}, Disc_loss: {loss_d/bi}")
            # Print the loss of the current iteration after `PRINT_EVERY` iterations.
            if (bi+1) % PRINT_EVERY == 0:
                print(f"[Epoch/Epochs] [{epoch+1}/{EPOCHS}], [Batch/Batches] [{bi+1}/{len(train_loader)}], Gen_loss: {bi_loss_g}, Disc_loss: {bi_loss_d}")
            global_batch_iter += 1

        # Save the generator model after the specified interval.
        if (epoch+1) % GEN_MODEL_SAVE_INTERVAL == 0:
            save_gen_model(
                epoch+1, generator, optim_g, criterion, 
                f"outputs_{DATASET}/generator_{epoch+1}.pth"
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
            {'disc_epoch_loss': epoch_loss_d, 'gen_epoch_loss': epoch_loss_g}, 
            epoch
        )
        epoch_end = time.time()
        
        print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}\n")
        print(f"Took {(epoch_end-epoch_start):.3f} seconds for epoch {epoch+1}")
        print('-'*50, end='\n')

    print('DONE TRAINING')
    # Save the generator model final time.
    if (epoch+1) % GEN_MODEL_SAVE_INTERVAL == 0:
        save_gen_model(
            EPOCHS, generator, optim_g, criterion, 
            f"outputs_{DATASET}/generator_final.pth"
        )


    # save the generated images as GIF file
    all_saved_image_paths = glob.glob(f"outputs_{DATASET}/gen_*.png")
    imgs = [Image.open(image_path) for image_path in all_saved_image_paths]
    imageio.mimsave(f"outputs_{DATASET}/generator_images.gif", imgs)

    # save epoch loss plot
    save_loss_plots(losses_g, losses_d, f"outputs_{DATASET}/epoch_loss.png")
    # save batch loss plot
    save_loss_plots(
        batch_losses_g, batch_losses_d, f"outputs_{DATASET}/batch_loss.png"
    )