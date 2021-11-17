import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


def return_data(
    BATCH_SIZE, data='FashionMNIST', 
    num_worders=0, image_size=32
):
    if data == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])
        train_data = datasets.MNIST(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
        )
        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=num_worders
        )
        return train_loader

    if data == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])
        train_data = datasets.FashionMNIST(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
        )
        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=num_worders
        )
        return train_loader

    if data == 'CIFAR10':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
        train_data = datasets.CIFAR10(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
        )
        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=num_worders
        )
        return train_loader

    if data == 'CELEBA':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
        train_data = datasets.ImageFolder(
        root='../input/data/celeba',
        transform=transform
        )
        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=num_worders
        )
        return train_loader

    if data == 'ABSTRACT_ART':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
        train_data = datasets.ImageFolder(
        root='../input/data/abstract_art_gallery/Abstract_gallery',
        transform=transform
        )
        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=num_worders
        )
        return train_loader