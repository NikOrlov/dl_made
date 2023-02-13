import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from src.config import (
    LOGGER,
    DATA_PATH,
    SEED,
    TRANSFORMS,
    NUM_CHANNELS,
    LEFT_SIDE,
    RIGHT_SIDE,
    DELTA,
    BS,
    NUM_EPOCHS,
    DEVICE,
    MODEL_PATH
)


from src.utils import train_step, test_step
from src.dataset import ImageDataset
from src.model import Model


if __name__ == "__main__":
    LOGGER.info("Start job")
    dataset = ImageDataset(
        path=DATA_PATH,
        transforms=TRANSFORMS,
        channels=NUM_CHANNELS,
        left_side=(LEFT_SIDE - DELTA),
        right_side=(RIGHT_SIDE + DELTA),
    )
    len_dataset = len(dataset)
    test_size = len_dataset // 5
    train_size = len_dataset - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    LOGGER.debug(
        f"Dataset size: {len_dataset}, train size: {train_size}, test size: {test_size}"
    )

    torch.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=BS, shuffle=False, num_workers=1)

    model = Model(in_channels=NUM_CHANNELS).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_step(model, train_loader, optimizer, criterion, NUM_EPOCHS)
    test_loss, test_cer = test_step(model, test_loader, criterion)

    torch.save(model.state_dict(), MODEL_PATH)
