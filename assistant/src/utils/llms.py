import os
import re

import chainlit as cl
import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import AppConfig
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


def create_hugging_face_model(config: AppConfig) -> tuple[ChatHuggingFace, dict]:
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
    model_dir = os.path.join("models", re.sub(r'[^a-zA-Z0-9\-/]', '_', model_name))
    model_dir = os.path.abspath(model_dir)

    # Check if model files exist
    model_path = os.path.join(model_dir)

    # Determine device and optimization settings based on performance mode
    performance_mode = config.system.models.local_lms.performance_mode
    device = torch.device("cuda" if performance_mode and torch.cuda.is_available() else "cpu")

    # Download and save model if it doesn't exist
    should_download = not os.path.exists(model_path)

    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    log.info(f"{'Downl' if should_download else 'L'}oading model to {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name if should_download else model_path,
        torch_dtype=torch.float16 if performance_mode and torch.cuda.is_available() else torch.float32,
        device_map="auto" if performance_mode and torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name if should_download else model_path)
    if device.type == "cpu":
        model = model.to(device)
    if should_download:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    # Set generation parameters
    generation_kwargs = {
        "do_sample": True,
        "temperature": float(settings["Temperature"]),
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": config.system.models.local_lms.max_new_tokens,
        "use_cache": True,
    }

    # Create the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        **generation_kwargs
    )

    # Create HuggingFacePipeline with streaming
    base_llm = HuggingFacePipeline(
        pipeline=pipe,
        model_id=model_path,
        model_kwargs={"stream": True}
    )

    # Create ChatHuggingFace with the base llm
    llm = ChatHuggingFace(
        llm=base_llm,
        human_prefix="Human:",
        ai_prefix="Assistant:"
    )

    return llm, settings
