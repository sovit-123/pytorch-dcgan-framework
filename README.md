# pytorch-dcgan-framework



## DCGAN Training on Different Datasets using PyTorch

* Currently, the generator generates 64x64 resolution images.



## Current Datasets Supported

* MNIST
* Fashion MNIST
* CIFAR10
* CELEBA 
* Abstract Art Gallery  



## Dataset Directory

The datasets are one folder back from the working project directory. Relative path from current project directory:

* `../input/data`

Following is the structure showing how all the datasets are arranged:

```
input
|───data
    ├───abstract_art_gallery
    │   ├───Abstract_gallery
    │   │   └───Abstract_gallery
    │   └───Abstract_gallery_2
    │       └───Abstract_gallery_2
    ├───celeba
    │   └───img_align_celeba
    ├───cifar-10-batches-py
    ├───FashionMNIST
    │   └───raw
    ├───MNIST
    │   └───raw
```

* MNIST, Fashion MNIST, and CIFAR10 data are directly downloaded from PyTorch `torchvision` module.
* To train on CelebA and Abstract Art Gallery dataset, you need to download them and arrange them proper directory first.
* [Download CelebA dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ).
  * Download the `img_align_celeba.zip` file.
* [Download Abstract Art Gallery dataset](https://www.kaggle.com/bryanb/abstract-art-gallery).



## Training Configurations

* The training configuration for MNIST and Fashion MNIST datasets are the same.

  * Just change the `DATASET` to `'MNIST'` or `'FashionMNIST'`. `N_CHANNELS` should be `1` for grayscale images.

    ```python
    import torch
    
    BATCH_SIZE = 128
    EPOCHS  = 100
    NUM_WORKERS = 4
    MULT_FACTOR = 1
    IMAGE_SIZE = 64*MULT_FACTOR
    # image channels
    N_CHANNELS = 1
    
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
    DATASET = 'MNIST'
    
    # for printing metrics
    PRINT_EVERY = 100
    
    # for optimizer
    BETA1 = 0.5
    BETA2 = 0.999
    LEARNING_RATE = 0.0002
    ```

* For CIFAR10 and other colored images datasets, change the `N_CHANNELS` to 3, RGB images.

  * ```python
    import torch
    
    BATCH_SIZE = 128
    EPOCHS  = 100
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
    DATASET = 'CIFAR10'
    
    # for printing metrics
    PRINT_EVERY = 100
    
    # for optimizer
    BETA1 = 0.5
    BETA2 = 0.999
    LEARNING_RATE = 0.0002
    ```

    
