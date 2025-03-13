from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize(config.dataset.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.ImageFolder(
        root=config.dataset.root,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True
    )



