import numpy as np
import torch
from config import DEVICE, LOGGER


def train_step(model, dataloader, opt, criterion, num_epochs) -> None:
    LOGGER.info("Start training!")
    model.train()
    for i in range(num_epochs):
        epoch_loss = []
        for image, label in dataloader:
            image, label = image.to(DEVICE), label.to(DEVICE)

            out = model(image)

            loss = criterion(out, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss.append(loss.item())

        epoch_loss_val = round(np.mean(epoch_loss), 3)
        LOGGER.info("Epoch %d loss: %f", i + 1, epoch_loss_val)


def test_step(model, dataloader, criterion, device=DEVICE) -> (int, int):
    LOGGER.info("Start validating!")
    model.eval()
    test_loss = []
    num_correct = 0
    num_labels = 0
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)

            out = model(image)
            loss = criterion(out, label)
            test_loss.append(loss.item())
            _, preds = torch.max(out, dim=1)
            num_correct += (preds == label).sum().item()
            num_labels += label.shape[0] * label.shape[-1]
        test_loss = round(np.mean(test_loss), 3)
        cer = round(1 - (num_correct / num_labels), 3)
    LOGGER.info("Test loss: %.4f, test CER: %.4f", test_loss, cer)
    return test_loss, cer
