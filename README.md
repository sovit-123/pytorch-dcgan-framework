# pytorch-dcgan-framework



## DCGAN Training on Different Datasets using PyTorch

* Currently, the generator generates 64x64 resolution images.



## Current Features/Supports
* You can resume training from any saved model.
* TensorBoard logging of loss graphs.
* Resuming training will also create new TensorBoard run where the old plots will be generated first, and then continue.
* The generator model is from the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434v2). I tried to replicate it as well as I could.
* The discriminator network follows all the rules from the [official paper](https://arxiv.org/abs/1511.06434v2) although there can be a few flexibilties to the size and depth of the network. Still, the core rules are all from the paper.
* Option to save GIFs of generated images after training ends (See `config.py`). **If trained for high number of epochs (>500), it will require a lot of RAM as all the saved images from the 500 epochs will be loaded directly to memory to create teh GIF**. Therefore provided option to turn it off. 




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



## Project Directory

```
.
├── config.py
├── datasets.py
├── models.py
├── outputs_ABSTRACT_ART
├── outputs_CELEBA
├── outputs_CIFAR10
├── outputs_FashionMNIST
├── outputs_MNIST
├── README.md
├── runs
├── train.py
└── utils.py
```



## Training Configurations

* The training configuration for MNIST and Fashion MNIST datasets are the same.

  * Just change the `DATASET` to `'MNIST'` or `'FashionMNIST'`. `N_CHANNELS` should be `1` for grayscale images.

    ```python
    import torch
    
    BATCH_SIZE = 128
    EPOCHS  = 50
    EPOCH_START = 0
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
    ```

* For CIFAR10 and other colored images datasets, change the `N_CHANNELS` to 3, RGB images.

  * ```python
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
    ```
    
    
