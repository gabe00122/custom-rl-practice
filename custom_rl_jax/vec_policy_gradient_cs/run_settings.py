import json
from typing import TypedDict

RunSettings = TypedDict(
    "RunSettings",
    {
        "git_hash": str,
        "seed": int,
        "total_steps": int,
        "env_name": str,
        "env_num": int,
        "discount": float,
        "root_hidden_layers": list[int],
        "actor_hidden_layers": list[int],
        "critic_hidden_layers": list[int],
        "actor_last_layer_scale": float,
        "critic_last_layer_scale": float,
        "learning_rate": float,
        "optimizer": str,
        "adam_beta": float,
        "weight_decay": float,
        "clip_norm": float,
        "actor_coef": float,
        "critic_coef": float,
        "entropy_coef": float,
    },
)


def save_settings(path, settings):
    with open(path, "w") as file:
        json.dump(settings, file, indent=2)


def load_settings(path):
    with open(path, "r") as file:
        settings = json.load(file)
    return settings
