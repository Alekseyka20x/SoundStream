import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def remove_target(config):
    if isinstance(config, dict):
        return {
            key: remove_target(value)
            for key, value in config.items()
            if key != "_target_"
        }
    if isinstance(config, list):
        return [remove_target(value) for value in config]
    return config


def main():
    pretrained_path = input("Path to checkpoint: ")
    output_path = input("Path to converted checkpoint: ")

    pretrained_path = str(pretrained_path)
    checkpoint = torch.load(pretrained_path, "cpu")

    generator_dict = checkpoint["model_processor"]["generator"]

    generator_config: OmegaConf = generator_dict["config"]["model"]
    if generator_config.get("target_sample_rate") is None:
        OmegaConf.set_struct(generator_config, False)
        generator_config.update({"target_sample_rate": 16000})
        OmegaConf.set_struct(generator_config, True)

    generator_state_dict = generator_dict["model"]

    model = instantiate(generator_config)
    model.load_state_dict(generator_state_dict)
    model.eval().cpu()

    raw_config = OmegaConf.to_container(generator_config, resolve=True)

    torch.save(
        {"state_dict": model.state_dict(), "config": remove_target(raw_config)},
        output_path,
    )


if __name__ == "__main__":
    main()
