import torch
from torch import nn
from torchvision import utils
from tqdm import tqdm
from pathlib import Path
import time
import logging
from torch.nn import functional as F
from training.loss import StyleGAN2Loss
from utils.dataloader import get_dataloader
from utils.init import init_generator, init_discriminator, init_schedulers, init_optimizers, \
    device_initializer, seed_initializer, amp_initializer
from utils.metrics import compute_ssim, compute_psnr, compute_mse, compute_lpips

LOGGER = logging.getLogger(__name__)


class GANTrainer:
    def __init__(self, config):

        self.config = config
        self.device = device_initializer(0, True)
        # 初始化配置
        self._init_config()
        seed_initializer()
        # 初始化组件
        self.train_loader = get_dataloader(config)
        self.gen = init_generator(config, self.device)
        self.disc = init_discriminator(config, self.device)
        self.opt_gen, self.opt_disc = init_optimizers(config, self.gen, self.disc)
        self.scheduler_gen, self.scheduler_disc = init_schedulers(config, self.opt_gen, self.opt_disc)
        self.criterion = self._get_loss()
        # 初始化固定噪声
        self._init_fixed_noise()
        # 训练状态
        self.current_epoch = 0
        self.avg_ssim = 0
        self.best_ssim = 0
        # 恢复训练
        self._resume_checkpoint()


    def _init_config(self):
        """解析配置文件"""
        # 数据配置

        self.img_channels = self.config.img_channels

        # 模型配置
        self.z_dim = self.config.z_dim
        self.w_dim = self.config.w_dim
        self.num_classes = self.config.num_classes

        # 训练配置

        self.epochs = self.config.epochs
        self.gen_lr = self.config.gen_lr
        self.disc_lr = self.config.disc_lr
        self.betas = tuple(self.config.betas)
        self.save_dir = Path(self.config.save_dir)
        self.save_interval = self.config.save_interval
        self.update_emas = self.config.update_emas
        self.truncation_psi = self.config.truncation_psi
        self.r1_gamma = self.config.r1_gamma
        self.r1_ssim = self.config.r1_ssim
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)


    def _get_loss(self):
        scaler = amp_initializer(amp=(self.device.type != "cpu"), device=self.device)
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
        if self.config.resume is None:
            return
        checkpoint = torch.load(self.config.resume)
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
            print("Starting training...")
            start_time = time.time()

            for epoch in range(self.current_epoch, self.epochs):
                self.before_train()
                self.training(epoch)
                self.after_training(epoch)
                self._evaluate(epoch)
                # 更新学习率

            print(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")

        except Exception as e:
            print(f"Training failed: {str(e)}")
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
            # The real sample label is 1, and the generated sample label is 0.
            real_label = torch.full((batch_size, 1), 1, dtype=inputs.dtype).to(device=self.device, non_blocking=True)
            fake_label = torch.full((batch_size, 1), 0, dtype=inputs.dtype).to(device=self.device, non_blocking=True)
            noise = torch.randn([batch_size, self.z_dim], device=self.device)
            conditional = F.one_hot(target, self.num_classes).to(self.device).float()
            self.disc.zero_grad()
            with torch.amp.autocast(self.device.type, enabled=self.device.type != "cpu"):
                inputs.requires_grad_(True)
                real_output = self.criterion.run_D(inputs, conditional)
            with torch.amp.autocast(self.device.type, enabled=self.device.type != "cpu"):
                fake = self.criterion.run_G(noise, conditional)
                fake_output = self.criterion.run_D(fake.detach(), conditional)
            total_loss = self.criterion.calc_D_loss(inputs, real_output, fake_output, real_label, fake_label)
            self.criterion.update(total_loss, self.opt_disc)
            self.gen.zero_grad()
            with torch.amp.autocast(self.device.type, enabled=self.device.type != "cpu"):
                fake_output = self.criterion.run_D(fake, conditional)
            g_loss = self.criterion.calc_G_loss(fake_output, real_label)
            ssim = compute_ssim(inputs, fake)
            mse = compute_mse(inputs, fake)
            psnr = compute_psnr(mse)
            lpips = compute_lpips(inputs, fake)
            g_loss_ssim = g_loss + self.r1_ssim * (1 - ssim) + 0.25 * mse.item() + lpips * 0.5
            self.criterion.update(g_loss_ssim, self.opt_gen)
            progress_bar.set_postfix({
                'D Loss': total_loss.item(),
                'G Loss': g_loss_ssim.item(),
                'ssim': ssim,
                "mse": mse.item(),
                "psnr":psnr,
                "lpips": lpips,
            })
            self.ssims.append(ssim)

    def after_training(self, epoch: int):
        self.scheduler_gen.step()
        self.scheduler_disc.step()
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
