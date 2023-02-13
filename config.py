import logging.config
import string
import torch
import torchvision


DATA_PATH = "samples/samples"
LOG_PATH = "training.log"
MODEL_PATH = "model.pth"

SEED = 42
LABELS = string.digits + string.ascii_lowercase
LABEL_TO_ID = {label: ids for ids, label in enumerate(LABELS)}
ID_TO_LABEL = {ids: label for label, ids in LABEL_TO_ID.items()}

LEFT_SIDE = 20
RIGHT_SIDE = 150
DELTA = 2
NUM_CHANNELS = 3


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORMS = torchvision.transforms.ToTensor()
NUM_EPOCHS = 50
BS = 32

log_conf = {
    "version": 1,
    "formatters": {
        "simple": {"format": "%(levelname)s\t%(message)s"},
        "extended": {"format": "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"},
    },
    "handlers": {
        "file_handler": {
            "level": "DEBUG",
            "filename": LOG_PATH,
            "formatter": "extended",
            "class": "logging.FileHandler",
        },
        "stream_handler": {
            "level": "INFO",
            "formatter": "simple",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "file_stream": {
            "level": "DEBUG",
            "handlers": ["file_handler", "stream_handler"],
        },
        "file": {"level": "DEBUG", "handlers": ["file_handler"]},
    },
}

logging.config.dictConfig(log_conf)
LOGGER = logging.getLogger("file_stream")
