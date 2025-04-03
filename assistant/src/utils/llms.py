import os
import re

import chainlit as cl
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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


def create_hugging_face_model(config: AppConfig) -> tuple[HuggingFacePipeline, dict]:
    """
    Creates an instance of HuggingFacePipeline based on app config and chat settings.
    The model is downloaded and cached locally in a models directory.

    :param config: the current application config.
    :return: tuple[HuggingFacePipeline, dict] containing the model and settings
    """
    settings = cl.user_session.get("chat_settings")

    if not "Model" in settings.keys():
        settings["Model"] = config.system.models.default.name
        log.debug(f"Using default model {settings['Model']}")

    if not "Temperature" in settings.keys():
        settings["Temperature"] = config.system.models.default.temperature
        log.debug(f"Using default temperature {settings['Temperature']}")

    # Create a directory name from the model name (replacing / with _)
    model_name = settings["Model"]
    model_dir = os.path.join("models", re.sub(r'[^a-zA-Z0-9\-/]', '_', model_name))
    model_dir = os.path.abspath(model_dir)

    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Check if model files exist
    model_path = os.path.join(model_dir, "model")
    tokenizer_path = os.path.join(model_dir, "tokenizer")

    # Determine device and optimization settings based on performance mode
    performance_mode = config.system.models.local_lms.performance_mode
    device = torch.device("cuda" if performance_mode and torch.cuda.is_available() else "cpu")

    # Default model kwargs and generation kwargs
    model_kwargs = {
        "torch_dtype": torch.float16 if performance_mode and torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if performance_mode and torch.cuda.is_available() else None,
        "low_cpu_mem_usage": True,
    }

    # Set generation parameters - these affect how the text is generated
    generation_kwargs = {
        "do_sample": performance_mode,
        "temperature": settings["Temperature"],
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": config.system.models.local_lms.max_new_tokens,
        "use_cache": True,
    }

    # Download and save model if it doesn't exist
    if not os.path.exists(model_path):
        log.info(f"Downloading model to {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        if device.type == "cpu":
            model = model.to(device)
        model.save_pretrained(model_path)
    else:
        log.info(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        if device.type == "cpu":
            model = model.to(device)

    # Download and save tokenizer if it doesn't exist
    if not os.path.exists(tokenizer_path):
        log.info(f"Downloading tokenizer to {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        log.info(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Create text generation pipeline with streaming capability
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # return_full_text=False,  # Don't return the prompt text in the output
        **generation_kwargs
    )

    # Create LangChain HuggingFacePipeline with proper configuration for streaming
    llm = HuggingFacePipeline(
        pipeline=pipe,
        model_id=model_name,
        model_kwargs={"stream": True}  # Enable streaming at the model level
    )

    return llm, settings
