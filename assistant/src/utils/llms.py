import chainlit as cl
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI

from config import AppConfig
from src.components.model import TransformersModel
from src.logger import log


def _set_defaults(config: AppConfig, settings: dict):
    if not "Model" in settings.keys():
        settings["Model"] = config.system.models.default.name
        log.debug(f"Using default deployment {settings['Model']}")

    if not "Temperature" in settings.keys():
        settings["Temperature"] = config.system.models.default.temperature
        log.debug(f"Using default temperature {settings['Temperature']}")


def create_chat_openai_model(config: AppConfig) -> tuple[ChatOpenAI, dict]:
    """
    Creates an instance of ChatOpenAI based on app config and chat settings.

    :param config: the current application config.

    :return: ChatOpenAI
    """
    settings = cl.user_session.get("chat_settings")
    _set_defaults(config, settings)

    settings["Streaming Enabled"] = True
    if settings["Model"] in ("o1", "o1-mini"):
        settings["Streaming Enabled"] = False
        settings["Temperature"] = 1

        log.debug(f"Enforcing o1 settings without streaming and temperature set to 1")

    return (
        ChatOpenAI(model=settings["Model"]),
        settings,
    )


def create_hugging_face_model(config: AppConfig) -> tuple[BaseLLM, dict]:
    """
    Creates an instance of ChatHuggingFace based on app config and chat settings.
    The model is downloaded and cached locally in a models directory.

    :param config: the current application config.
    :return: tuple[ChatHuggingFace, dict] containing the model and settings
    """
    settings = cl.user_session.get("chat_settings")
    _set_defaults(config, settings)

    # Create a directory name from the model name (replacing / with _)
    model_name = settings["Model"]
    # Set generation parameters
    generation_kwargs = config.system.models.local_lms.generation_kwargs

    llm = TransformersModel(
        model_name=model_name,
        generation_kwargs=generation_kwargs,
    )

    return llm, settings
