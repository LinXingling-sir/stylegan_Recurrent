import yaml
from omegaconf import OmegaConf

from training.training_loop import GANTrainer

if __name__ == "__main__":
    # 配置示例（实际应该使用OmegaConf加载配置文件）
    with open('./config/setting.yaml', 'r', encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    config = OmegaConf.create(config_data)
    trainer = GANTrainer(config)
    trainer.train()
