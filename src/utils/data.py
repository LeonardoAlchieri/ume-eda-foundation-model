from torch.utils.data import DataLoader

def get_dataloder(dataset, batch_size: int, num_workers: int = 4, shuffle: bool = True, ):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )