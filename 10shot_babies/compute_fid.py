import argparse
import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
import torchvision.models as models
from tqdm import tqdm
# import viz
from copy import deepcopy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from glob import glob
from PIL import Image
import pandas as pd
import  pytorch_fid_wrapper as pfw

pfw.set_config(batch_size=50, dims = 2048, device='cuda')

from NLI_model import Generator
from model import Discriminator
from NLI_model import NLI, LatentLearner

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
                (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_loss(cnn, normalization_mean, normalization_std, target_img, style_layers, device):

    style_losses = []
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            target_feature = model(target_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses


def set_weights(layer_shapes, pred_weights):
    idx, params = 0, []
    for layer in layer_shapes:
        layer_params = []
        for shape in layer:
            offset = np.prod(shape)
            layer_params.append(pred_weights[:, :, idx:(idx + offset)].reshape((-1, 14, *shape)))
            idx += offset
        params.append(layer_params)
    return params


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=0)
    parser.add_argument("--img_freq", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--adv_bs", type=int, default=8)
    parser.add_argument("--sty_bs", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=501)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--ckpt", type=str, default='../checkpoint/550000_backup.pt')
    parser.add_argument("--s_ds", type=str, default='ffhq')
    parser.add_argument("--t_ds", type=str, default='babies')

    args = parser.parse_args()

    # torch.manual_seed(1)
    # random.seed(1)
    # np.random.seed(1)

    n_gpu = 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    s_ds = args.s_ds

    t_ds = args.t_ds

    model_path = os.path.join(f'exp/checkpoints/{args.t_ds}')

    layer_shapes = [
        [(256, 1024), (256,)],
        # [(256, 256), (256,)],
        [(512, 256), (512,)]
    ]
    # nw = (1024 * 256 + 256) + (256 * 256 + 256) + (256 * 512 + 512)
    nw = (1024 * 256 + 256) + (256 * 512 + 512)

    hyper_net = nn.Sequential(
        nn.Linear(1024, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, nw),
        nn.Tanh()
    ).to(device)
    
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    d = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema.eval()
    # accumulate(g_ema, generator, 0)

    hyper_optim = optim.Adam(
        hyper_net.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        d.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        g_ema.load_state_dict(ckpt_source["g_ema"], strict=False)
        d.load_state_dict(ckpt_source["d"], strict=False)

    if args.distributed:
        g_ema = nn.parallel.DataParallel(g_ema)

    x_real_path = glob(f'../data/real_{t_ds}/*.png')
    real_imgs = []
    for img_path in x_real_path:
        img = np.array(Image.open(img_path))
        real_imgs.append(np.expand_dims(img, 0))
    real_imgs = np.concatenate(real_imgs)
    real_imgs = (real_imgs - np.min(real_imgs)) / (np.max(real_imgs) - np.min(real_imgs))
    if real_imgs.shape[-1] == 1:
        real_imgs = np.repeat(real_imgs, repeats=3, axis=3)
    real_imgs = np.transpose(real_imgs, (0, 3, 1, 2))
    real_imgs = torch.Tensor(real_imgs)
    real_m, real_s = pfw.get_stats(real_imgs)
    fid_buf = []
    steps_buf = []
    for step in range(args.start, args.stop, 1):
        w_target = torch.randn([args.n_samples, 14, 512]).to(device)
        w_noise = torch.randn_like(w_target)
        hyper_net.load_state_dict(torch.load(f'{model_path}/hyper-{step}.pt')['hyper_net'], strict=False)
        imsave_path = os.path.join(f'exp/inferred/{args.t_ds}', str(step))

        if not os.path.exists(imsave_path):
            os.makedirs(imsave_path)
        
        with torch.no_grad():
            for idx, (t, n) in tqdm(enumerate(zip(w_target, w_noise)), total=args.n_samples):
                weights = hyper_net(torch.cat([t.unsqueeze(0), n.unsqueeze(0)], 2))
                params = set_weights(layer_shapes=layer_shapes, pred_weights=weights)
                w_mixed = LatentLearner(torch.cat([t.unsqueeze(0), n.unsqueeze(0)], 2), params)
                interpolated_sample, _ = g_ema([w_mixed], input_is_latent=True, return_feats=False)
                utils.save_image(
                    interpolated_sample.detach().clamp_(min=-1, max=1),
                    f"%s/generated_{step}_{idx}.png" % (imsave_path),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                
        fake_imgs = []
        x_fake_path = glob(f'{imsave_path}/*.png')
        for img_path in x_fake_path:
            img = np.array(Image.open(img_path))
            fake_imgs.append(np.expand_dims(img, 0))
        fake_imgs = np.concatenate(fake_imgs)
        fake_imgs = (fake_imgs - np.min(fake_imgs)) / (np.max(fake_imgs) - np.min(fake_imgs))
        if len(fake_imgs.shape) == 3:
            fake_imgs = fake_imgs[..., np.newaxis]
            fake_imgs = np.repeat(fake_imgs, repeats=3, axis=3)
        fake_imgs = np.transpose(fake_imgs, (0, 3, 1, 2))
        fake_imgs = torch.Tensor(fake_imgs)

        fid_score = pfw.fid(fake_imgs, real_m=real_m, real_s=real_s)
        
        fid_buf.append(fid_score)
        steps_buf.append(step)

        print(f'Step: {step}, FID: {fid_score:.4f}\n')

df = pd.DataFrame({'step': steps_buf, 'fid': fid_buf})
df.to_csv(f'./fid_stats_{t_ds}_{args.start}_{args.stop}.csv')
