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
