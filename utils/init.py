import logging
import random

import coloredlogs
import numpy as np
import torch
from torch import optim, GradScaler

from model import Generator, Discriminator

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def device_initializer(device_id=0, is_train=False):
    """
    This function initializes the running device information when the program runs for the first time
    [Warn] This project will no longer support CPU training after v1.1.2
    :param device_id: Device id
    :param is_train: Whether to train mode
    :return: cpu or cuda
    """
    logger.info(msg="Init program, it is checking the basic device setting.")
    device_dict = {}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set device with custom setting
        device = torch.device("cuda", device_id)
        is_init = torch.cuda.is_initialized()
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(device=device)
        device_cap = torch.cuda.get_device_capability(device=device)
        device_prop = torch.cuda.get_device_properties(device=device)
        device_dict["device_id"] = device_id
        device_dict["is_init"] = is_init
        device_dict["device_count"] = device_count
        device_dict["device_name"] = device_name
        device_dict["device_cap"] = device_cap
        device_dict["device_prop"] = device_prop
        logger.info(msg=device_dict)
        return device
    else:
        logger.warning(msg="This project will no longer support CPU training after version 1.1.2")
        if is_train:
            raise NotImplementedError("CPU training is no longer supported after version 1.1.2")
        else:
            # Generate or test mode
            logger.warning(msg="Warning: The device is using cpu, the device would slow down the models running speed.")
            return torch.device(device="cpu")


def seed_initializer(seed_id=0):
    """
    Initialize the seed
    :param seed_id: The seed id
    :return: None
    """
    torch.manual_seed(seed_id)
    torch.cuda.manual_seed_all(seed_id)
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(msg=f"The seed is initialized, and the seed ID is {seed_id}.")


def init_generator(config, device):
    gen = Generator(
        z_dim=config.z_dim,
        c_dim=config.num_classes,
        w_dim=config.w_dim,
        img_resolution=config.img_size,
        img_channels=config.img_channels
    ).to(device)
    return gen


def init_discriminator(config, device):
    disc = Discriminator(
        c_dim=config.num_classes,
        img_resolution=config.img_size,
        img_channels=config.img_channels
    ).to(device)
    return disc


def amp_initializer(amp, device):
    """
    Initialize automatic mixed precision
    :param amp: Enable automatic mixed precision
    :param device: GPU or CPU
    :return: scaler
    """
    if amp:
        logger.info(msg=f"[{device}]: Automatic mixed precision training.")
    else:
        logger.info(msg=f"[{device}]: Normal training.")
    # Used to scale gradients to prevent overflow
    return GradScaler(enabled=amp)




def init_optimizers(config, gen, disc):
    """初始化优化器"""
    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.gen_lr,
        betas=tuple(config.betas),
        eps=float(config.eps),
    )

    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.disc_lr,
        betas=tuple(config.betas),
        eps=float(config.eps),
    )
    return opt_gen, opt_disc


def init_schedulers(config, opt_gen, opt_disc):
    """初始化学习率调度器"""
    scheduler_gen = optim.lr_scheduler.StepLR(
        opt_gen,
        step_size=config.lr_step,
        gamma=config.lr_gamma
    )

    scheduler_disc = optim.lr_scheduler.StepLR(
        opt_disc,
        step_size=config.lr_step,
        gamma=config.lr_gamma
    )

    return scheduler_gen, scheduler_disc
