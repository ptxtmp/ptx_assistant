import os
from typing import Any

import chainlit as cl
import yaml

from src.components.chat import ChatProfileConfig
from src.utils.common import flatten_list, load_class, DotDict


class AppConfig(DotDict):
    """dot.notation access to dictionary keys as attributes"""

    @property
    def debug(self):
        return self.system.chainlit.debug

    @property
    def app_name(self):
        return self.system.chainlit.app_name

    def get_chat_profile(self, chat_profile: str, key: str):
        return self.get_chat_profile_dict(key=key)[chat_profile]

    def get_chat_handler_class(self, chat_profile: Any, key: str = None):
        if isinstance(chat_profile, str):
            chat_profile = self.get_chat_profile(chat_profile, key)

        return load_class(f"...profiles.{chat_profile.handler_class}")

    def get_chat_profile_settings(self, chat_profile: Any, key: str = None):
        if isinstance(chat_profile, str):
            chat_profile = self.get_chat_profile(chat_profile, key)

        return cl.ChatSettings(chat_profile.settings.input_widgets())

    async def sync_chat_profile_settings(self, chat_profile: str, settings: dict):
        setting_widgets = self.get_chat_profile_settings(chat_profile=chat_profile, key="name")

        for inp in setting_widgets.inputs:
            inp.initial = settings[inp.id]

        return await setting_widgets.send()

    def get_chat_profile_dict(self, key: str = "key"):
        chat_profiles = {}

        for profile_key, profile in self.chat_profiles.items():
            if isinstance(profile.settings, list):
                profile.settings = flatten_list(profile.settings)

            if isinstance(profile.starters, list):
                profile.starters = flatten_list(profile.starters)

            if key == "key":
                chat_profiles[profile_key] = ChatProfileConfig.model_validate(profile)
            elif isinstance(profile[key], str) and key in profile:
                chat_profiles[profile[key]] = ChatProfileConfig.model_validate(profile)
            else:
                raise ValueError(
                    f"Requested key '{key}' for the output chat profile dict is either None or not a string"
                )

        return AppConfig.from_dict(data=chat_profiles)

    def get_cl_chat_profile_list(self):
        return [
            profile.to_cl_chat_profile()
            for key, profile in self.get_chat_profile_dict(key="key").items()
            if profile.active and os.getenv(f"{key.upper()}_ACTIVE", "true").lower() in ("true", "yes")
        ]

    @staticmethod
    def _is_env_conf(conf):
        return len(conf.keys()) == 2 and "type" in conf and "key" in conf and conf["type"] == "env"

    @classmethod
    def load_from_yaml(cls, config_file_path: str):
        with open(config_file_path, "r", encoding="utf8") as fp:
            data = yaml.safe_load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any):
        if isinstance(data, dict) and cls._is_env_conf(data):
            return os.getenv(data["key"])
        else:
            return super(AppConfig, cls).from_dict(data)
