from omegaconf import OmegaConf


def get_stage1_params(config):
    if "stage1" in config:
        return config.stage1.params
    return config.model.params


def get_dit_params(config):
    if "dit" in config:
        return OmegaConf.merge({"attn_drop": 0.1, "mlp_drop": 0.1}, config.dit.params)
    return config.model.params


def get_dit_scheduler(config):
    if "dit" in config and "scheduler" in config.dit:
        return config.dit.scheduler
    return config.scheduler


def with_defaults(config, section, defaults):
    values = config.get(section, {})
    return OmegaConf.merge(defaults, values)


def validate_tgca_base_config(config):
    """Reject unconditional-only options before loading a TGCA checkpoint."""
    if config.get("scheduler_type", "ddpm") != "ddpm":
        raise ValueError("TGCA currently requires a DDPM-trained base; flow matching is unconditional-only")
    if get_dit_params(config).get("self_conditioning", False):
        raise ValueError("TGCA does not support a self-conditioned base model")
    if config.get("training", {}).get("normalize_latents", False):
        raise ValueError("TGCA does not support normalized-latent base models")
