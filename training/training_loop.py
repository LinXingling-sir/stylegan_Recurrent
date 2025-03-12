import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from tqdm import tqdm
from pathlib import Path
import time
import logging
from torch.nn import functional as F

from model.Discriminator import Discriminator
from model.stylegan2 import Generator
from training.loss import StyleGAN2Loss
from training.utils import compute_ssim

LOGGER = logging.getLogger(__name__)


class GANTrainer:
    def __init__(self, config, device):


        self.config = config
        self.device = device
        # 初始化配置
        self._init_config()
        # 初始化组件
        self.train_loader = self._get_dataloader()
        self.gen, self.disc = self._init_models()
        self.opt_gen, self.opt_disc = self._init_optimizers()
        self.scheduler_gen, self.scheduler_disc = self._init_schedulers()
        self.criterion = self._get_loss()
        # 训练状态
        self.current_epoch = 0
        self.avg_ssim = 0
        # 初始化固定噪声
        self._init_fixed_noise()
        # 恢复训练
        self._resume_checkpoint()

    def _init_config(self):
        """解析配置文件"""
        # 数据配置
        data_cfg = self.config.dataset
        self.data_dir = data_cfg.root
        self.img_size = data_cfg.img_size
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers
        self.img_channels = data_cfg.img_channels

        # 模型配置
        model_cfg = self.config.model
        self.z_dim = model_cfg.z_dim
        self.w_dim = model_cfg.w_dim
        self.num_classes = model_cfg.num_classes

        # 训练配置
        train_cfg = self.config.train
        self.epochs = train_cfg.epochs
        self.gen_lr = train_cfg.gen_lr
        self.disc_lr = train_cfg.disc_lr
        self.betas = tuple(train_cfg.betas)
        self.save_dir = Path(train_cfg.save_dir)
        self.save_interval = train_cfg.save_interval
        self.update_emas = train_cfg.update_emas
        self.truncation_psi = train_cfg.truncation_psi
        self.r1_gamma = train_cfg.r1_gamma
        self.r1_ssim = train_cfg.r1_ssim
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataloader(self) -> DataLoader:
        """创建数据加载器"""
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _init_models(self) -> tuple[nn.Module, nn.Module]:
        """初始化生成器和判别器"""
        gen = Generator(
            z_dim=self.z_dim,
            c_dim=self.num_classes,
            w_dim=self.w_dim,
            img_resolution=self.img_size,
            img_channels=self.img_channels
        ).to(self.device)

        disc = Discriminator(
            c_dim=self.num_classes,
            img_resolution=self.img_size,
            img_channels=self.img_channels
        ).to(self.device)

        # 加载预训练权重
        if self.config.model.get("gen_weights"):
            gen.load_state_dict(torch.load(self.config.model.gen_weights))
        if self.config.model.get("disc_weights"):
            disc.load_state_dict(torch.load(self.config.model.disc_weights))
        self.pl_mean = torch.zeros([], device=self.device)
        return gen, disc

    def _init_optimizers(self) -> tuple[optim.Optimizer, optim.Optimizer]:
        """初始化优化器"""
        opt_gen = optim.Adam(
            self.gen.parameters(),
            lr=self.gen_lr,
            betas=self.betas,
            eps=1e-8,
        )

        opt_disc = optim.Adam(
            self.disc.parameters(),
            lr=self.disc_lr,
            betas=self.betas,
            eps=1e-8,
        )
        self.best_ssim = 0
        return opt_gen, opt_disc

    def _init_schedulers(self):
        """初始化学习率调度器"""
        scheduler_gen = optim.lr_scheduler.StepLR(
            self.opt_gen,
            step_size=self.config.train.lr_step,
            gamma=self.config.train.lr_gamma
        )

        scheduler_disc = optim.lr_scheduler.StepLR(
            self.opt_disc,
            step_size=self.config.train.lr_step,
            gamma=self.config.train.lr_gamma
        )

        return scheduler_gen, scheduler_disc

    def _get_loss(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type != "cpu")
        """获取损失函数"""
        return StyleGAN2Loss(self.device, self.gen, self.disc, scaler, self.r1_gamma)

    def _init_fixed_noise(self):
        """初始化固定噪声用于可视化"""
        self.fixed_z = torch.randn(self.num_classes, self.z_dim, device=self.device)
        self.fixed_labels = torch.arange(0, self.num_classes) % self.num_classes
        self.fixed_c = nn.functional.one_hot(
            self.fixed_labels, self.num_classes
        ).float().to(self.device)

    def _resume_checkpoint(self):
        """恢复训练检查点"""
        if not self.config.train.get("resume"):
            return
        checkpoint = torch.load(self.config.train.resume)
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.disc.load_state_dict(checkpoint['disc_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['gen_optimizer'])
        self.opt_disc.load_state_dict(checkpoint['disc_optimizer'])
        self.best_ssim = checkpoint['ssim']
        self.current_epoch = checkpoint['epoch'] + 1
        LOGGER.info(f"Resumed from epoch {self.current_epoch}")

    def train(self):
        """主训练循环"""
        try:
            LOGGER.info("Starting training...")
            start_time = time.time()

            for epoch in range(self.current_epoch, self.epochs):
                self.before_train()
                self.training(epoch)
                self.after_training(epoch)
                self._evaluate(epoch)
                # 更新学习率

            LOGGER.info(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")

        except Exception as e:
            LOGGER.error(f"Training failed: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()

    def before_train(self):
        self.gen.train()
        self.disc.train()
        self.ssims = []

    def training(self, epoch: int):
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for i, (inputs, target) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            batch_size = inputs.size(0)
            noise = torch.randn([batch_size, self.z_dim], device=self.device)
            conditional = F.one_hot(target, self.num_classes).to(self.device).float()
            self.disc.zero_grad()
            with torch.amp.autocast(self.device, enabled=self.device.type != "cpu"):
                inputs.requires_grad_(True)
                real_output = self.criterion.run_D(inputs, conditional)
            with torch.amp.autocast(self.device, enabled=self.device.type != "cpu"):
                fake = self.criterion.run_G(noise, conditional)
                fake_output = self.criterion.run_D(fake.detach(), conditional)
            total_loss = self.criterion.calc_D_loss(inputs, real_output, fake_output)
            self.criterion.update(total_loss, self.opt_disc)
            self.gen.zero_grad()
            with torch.amp.autocast(self.device, enabled=self.device.type != "cpu"):
                fake_output = self.criterion.run_D(fake, conditional)
            g_loss = self.criterion.calc_G_loss(fake_output)
            ssim = compute_ssim(inputs, fake)
            g_loss_ssim = g_loss + self.r1_ssim * (1 - ssim)
            self.criterion.update(g_loss_ssim, self.opt_gen)
            progress_bar.set_postfix({
                'D Loss': total_loss.item(),
                'G Loss': g_loss_ssim.item(),
                'ssim': ssim,
            })
            self.ssims.append(ssim)

    def after_training(self, epoch: int):
        self.criterion.tostep(self.scheduler_gen, self.scheduler_disc)
        avg_ssim = sum(self.ssims) / len(self.ssims)
        if (epoch + 1) % self.save_interval != 0:
            return
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
            state = {
                'epoch': epoch,
                'gen_state_dict': self.gen.state_dict(),
                'disc_state_dict': self.disc.state_dict(),
                'gen_optimizer': self.opt_gen.state_dict(),
                'disc_optimizer': self.opt_disc.state_dict(),
                'ssim': avg_ssim,
            }
            torch.save(state, f"{self.save_dir}/checkpoint_epoch{epoch + 1}.pth")
            LOGGER.info(f"Saved checkpoint at epoch {epoch + 1}")

    def _evaluate(self, epoch: int):
        """评估和可视化"""
        self.gen.eval()
        with torch.no_grad():
            sample_imgs = self.gen(self.fixed_z, self.fixed_c)
            utils.save_image(
                sample_imgs,
                self.save_dir / f"epoch_{epoch + 1}.png",
                nrow=5,
                normalize=True
            )
