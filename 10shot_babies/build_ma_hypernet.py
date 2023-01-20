import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
import torchvision.models as models
from tqdm import tqdm
from copy import deepcopy
from glob import glob
from PIL import Image
from glob import glob
from natsort import natsorted
from tqdm import tqdm

from NLI_model import Generator
from model import Discriminator


MOVING_AVERAGE_DECAY = 0.1


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

    return ma_model


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def get_encoder(encoder_path):
    nw = (1024 * 256 + 256) + (256 * 256 + 256) + (256 * 512 + 512)
    target_encoder = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, nw),
        nn.Tanh()
    )
    target_encoder.load_state_dict(torch.load(encoder_path), strict=False)
    set_requires_grad(target_encoder, False)
    return target_encoder


online_latent_learners = natsorted(glob('exp/checkpoints/sunglass/*.pt'))
target_latent_learner = get_encoder(online_latent_learners[11])

target_ema_updater = EMA(MOVING_AVERAGE_DECAY)

for online_latent_learner in tqdm(online_latent_learners[12:]):
    target_latent_learner = update_moving_average(target_ema_updater, target_latent_learner, get_encoder(online_latent_learner))

torch.save(target_latent_learner.state_dict(), f'exp/checkpoints/sunglass/hypernet-ma.pt')
