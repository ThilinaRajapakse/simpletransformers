def sweep_config_to_sweep_values(sweep_config):
    """
    Converts an instance of wandb.Config to plain values map.

    wandb.Config varies across versions quite significantly,
    so we use the `keys` method that works consistently.
    """

    return {key: sweep_config[key] for key in sweep_config.keys()}
