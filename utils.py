import numpy as np
import torch
from config import DEVICE, LOGGER


def train_step(model, dataloader, opt, criterion, num_epochs, device=DEVICE) -> None:
    LOGGER.info('Start training!')
    model.train()
    for i in range(num_epochs):
        epoch_loss = []
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            out = model(X)

            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss.append(loss.item())

        epoch_loss_val = round(np.mean(epoch_loss), 3)
        LOGGER.info(f'Epoch {i + 1}) loss: {epoch_loss_val}')


def test_step(model, dataloader, criterion, device=DEVICE) -> (int, int):
    LOGGER.info('Start validating!')
    model.eval()
    test_loss = []
    num_correct = 0
    num_labels = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            out = model(X)
            loss = criterion(out, y)
            test_loss.append(loss.item())
            _, preds = torch.max(out, dim=1)
            num_correct += (preds == y).sum().item()
            num_labels += y.shape[0] * y.shape[-1]
        test_loss = round(np.mean(test_loss), 3)
        cer = round(1 - (num_correct / num_labels), 3)
    LOGGER.info(f'Test loss: {test_loss}, test CER: {cer}')
    return test_loss, cer
