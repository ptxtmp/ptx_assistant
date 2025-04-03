import chainlit as cl
from langchain_openai import ChatOpenAI

from config import AppConfig
from src.logger import log


def create_chat_openai_model(config: AppConfig) -> tuple[ChatOpenAI, dict]:
    """
    Creates an instance of ChatOpenAI based on app config and chat settings.

    :param config: the current application config.

    :return: ChatOpenAI
    """

    settings = cl.user_session.get("chat_settings")

    if not "Model" in settings.keys():
        settings["Model"] = config.system.models.default.name
        log.debug(f"Using default deployment {settings['Model']}")

    if not "Temperature" in settings.keys():
        settings["Temperature"] = config.system.models.default.temperature
        log.debug(f"Using default temperature {settings['Temperature']}")

    settings["Streaming Enabled"] = True
    if settings["Model"] in ("o1", "o1-mini"):
        settings["Streaming Enabled"] = False
        settings["Temperature"] = 1

        log.debug(f"Enforcing o1 settings without streaming and temperature set to 1")

    return (
        ChatOpenAI(model_name=settings["Model"]),
        settings,
    )
