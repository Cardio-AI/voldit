from omegaconf import OmegaConf


def get_stage1_params(config):
    if "stage1" in config:
        return config.stage1.params
    return config.model.params


def get_dit_params(config):
    if "dit" in config:
        return config.dit.params
    return config.model.params


def get_dit_scheduler(config):
    if "dit" in config and "scheduler" in config.dit:
        return config.dit.scheduler
    return config.scheduler


def with_defaults(config, section, defaults):
    values = config.get(section, {})
    return OmegaConf.merge(defaults, values)
