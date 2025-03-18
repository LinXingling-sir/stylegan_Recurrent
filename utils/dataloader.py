from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.ImageFolder(
        root=config.root,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )



