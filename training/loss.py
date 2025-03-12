from torch.nn import functional as F
import torch


class StyleGAN2Loss:
    def __init__(self, device, G, D, scaler, r1_gamma=10):
        super().__init__()
        self.scaler = scaler
        self.device = device
        self.G = G
        self.D = D
        self.r1_gamma = r1_gamma

    def run_G(self, z, c):
        fake = self.G(z, c)
        return fake

    def run_D(self, img, c):
        output = self.D(img, c)
        return output

    def calc_D_loss(self, inputs, real_output, fake_output):
        d_loss_real = F.softplus(-real_output).mean()
        gradients = torch.autograd.grad(
            outputs=real_output.sum(),  # 对标量输出求和计算梯度
            inputs=inputs,
            create_graph=True,  # 保留计算图以支持二阶导数
            only_inputs=True
        )[0]
        r1_penalty = gradients.square().sum(dim=(1, 2, 3)).mean()
        r1_loss = self.r1_gamma * r1_penalty
        d_loss_fake = F.softplus(fake_output).mean()
        total_loss = d_loss_real + d_loss_fake + r1_loss
        return total_loss

    def calc_G_loss(self, fake_output):
        g_loss = F.softplus(-fake_output).mean()
        return g_loss

    def update(self, loss, opt):
        self.scaler.scale(loss).backward()
        self.scaler.step(opt)
        self.scaler.update()

    def tostep(self,gen,disc):
        gen.setp()
        disc.setp()
