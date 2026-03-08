import os
import torch
import yaml

from src.models import df_iqa_cnn
from src.utils.logging import get_logger
from src.utils.timer import PROGRAM_START_TIME


def load_config(config_path="./config.yaml"):
    with open(config_path, mode='r') as f:
        config = yaml.safe_load(f)

    device = config.get("device", "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "mps" and not torch.mps.is_available():
        device = "cpu"

    config["device"] = device

    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    return config


if __name__ == "__main__":
    config = load_config()

    log_dir = config["log_dir"]
    logger, log_file = get_logger(log_dir=log_dir)
    logger.info(f"日志文件：{os.path.basename(log_file)}")

    device = config["device"]
    logger.info(f"运行设备：{device}")

    model_name = config["model"]
    model_mode = config["mode"]
    logger.info(f"当前模型：{model_name} | 模式：{model_mode}")
    if model_name == "df_iqa_cnn":
        module = df_iqa_cnn
    else:
        logger.error(f"模型未找到：{model_name}")
        exit(1)

    if model_mode == "train":
        save_dir = config["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_name, PROGRAM_START_TIME.strftime("%Y%m%d_%H%M%S"))
        logger.info(f"模型保存：{save_path}/ \n")
        module.train(config, logger, save_path)
    elif model_mode == "test":
        models_dir = config["models_dir"]
        logger.info(f"加载模型：{models_dir} \n")
        module.test(models_dir, config, logger)
    else:
        logger.error(f"未知的模式：{model_mode} 可选模式：train, test")