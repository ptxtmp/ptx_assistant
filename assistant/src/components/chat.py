from datetime import datetime
from typing import List, Optional

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from pydantic import RootModel

from src.components.solution import AutoFormattedStep
from src.utils.common import load_class, flatten_list


def load_input_widget(widget_type: str, options: dict):
    return load_class(f"chainlit.input_widget.{widget_type}")(**options)


def load_input_widgets(widgets: List[dict]):
    flattened_widgets = flatten_list(widgets)
    return [load_input_widget(widget["type"], widget["options"]) for widget in flattened_widgets]


class ChatProfileHandler:
    """
    A base class for handling chat profiles in a Chainlit application.

    This class defines the interface for chat profile handlers, which are responsible
    for managing the lifecycle of a chat session within a specific profile. It is designed
    to work with Chainlit's architecture, providing hooks for initialization and message processing.

    Subclasses should implement the `start` and `main` methods to define the specific
    behavior for each chat profile.
    """

    @staticmethod
    async def start(config, settings: cl.ChatSettings):
        """
        Initializes a new chat session for the profile.

        This method is called when a user starts a new chat with the profile. It should
        set up any necessary resources or state for the chat session.

        Both the method parameters are automatically sent to the class and therefore
        SHOULD NOT be interfered with.

        Args:
            config: The configuration object for the chat profile.
            settings (cl.ChatSettings): The settings object containing user-defined
                                        parameters for the chat session.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """

        # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Following is the example of how you can get the     #
        # basic components to start implementing this method  #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # Initialize the settings you need in the profile like this
        # # This is a mandatory step if you need control for settings in your profile
        # settings = await settings.send()
        #
        # # This is how you can have access to ENV variables in your profile
        # azure = config.system.azure
        # chainlit = config.system.chainlit
        #
        # # This is how you can access chat_profile-specific settings from config
        # profile_config = config.chat_profiles.{{profile_name}}
        #
        # # This is how you get the user info from the Chainlit Auth
        # user = cl.user_session.get("user")
        #
        # # Access the ENV variables like this
        # # For more info look into the `chat.config.yaml` file
        # openai_api_type=azure.openai.api_type,
        # openai_api_version=azure.openai.api_version,
        # openai_api_key=azure.openai.api_key,
        # azure_endpoint=azure.openai.base_url

        raise NotImplementedError

    @staticmethod
    async def main(config, message: cl.Message):
        """
        Processes incoming messages in the chat session.

        This method is called for each message sent by the user after the chat session
        has been initialized. It should contain the logic for generating responses and
        managing the conversation flow.

        Both the method parameters are automatically sent to the class and therefore
        SHOULD NOT be interfered with.

        Args:
            config: The configuration object for the chat profile.
            message (cl.Message): The incoming message object from the user.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Most examples show in the start method works here as well #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        raise NotImplementedError

    @staticmethod
    async def settings_update(config, settings: dict):
        """
        Processes change in settings.

        This method is called for each settings updated by the user

        The method parameters is automatically sent to the class and therefore
        SHOULD NOT be interfered with.

        Args:
            config: The configuration object for the chat profile.
            settings: The newly set settings
        """
        pass

    @staticmethod
    async def stop(config):
        """
        Handles the interruption of AI response generation when the user clicks the stop button.

        This method is called when the user manually stops AI response generation by clicking
        the stop button in the chat interface. It should contain logic to gracefully
        interrupt the current operation, free up resources, etc.

        Args:
            config: The configuration object for the chat profile.
        """
        pass

    @staticmethod
    async def end(config):
        """
        Handles the end of the chat session.

        This method is called when the user manually ends the chat session. It should
        contain logic to gracefully end the chat session, free up resources, etc.

        Args:
            config: The configuration object for the chat profile.
        """
        pass

    @staticmethod
    async def start_init_profile_step():
        profile_init_step = AutoFormattedStep(name="InitializeChatProfile", type="tool", language=None)
        await profile_init_step.send()

        # noinspection PyArgumentList
        profile_init_step.output = "`Initializing profile ...`"
        await profile_init_step.update()

        cl.user_session.set("profile_init_step", profile_init_step)

    @staticmethod
    async def end_init_profile_step():
        profile_init_step: AutoFormattedStep = cl.user_session.get("profile_init_step")

        # noinspection PyArgumentList
        profile_init_step.output = "`Chat profile initialization is complete!`"
        await profile_init_step.update()

        await cl.sleep(1)
        await profile_init_step.remove()

    @staticmethod
    async def ensure_user_session_values(config, keys: list[str], max_retries: int = 5):
        profile_init_step: AutoFormattedStep = cl.user_session.get("profile_init_step")

        params = {key: cl.user_session.get(key) for key in keys}

        if any(value is None for value in params.values()):
            # noinspection PyArgumentList
            profile_init_step.output = "`Chat profile initialization is still in progress ...`"
            await profile_init_step.update()

            counter = 0
            while True:
                params = {key: cl.user_session.get(key) for key in keys}
                if any(value is None for value in params.values()):
                    counter += 1
                    if counter > max_retries:
                        msg = "`Chat profile initialization could not complete. Please try again later!`"
                        # noinspection PyArgumentList
                        profile_init_step.output = msg
                        await profile_init_step.update()
                        await ChatProfileHandler.end_init_profile_step()
                        await cl.Message(content=msg).send()
                        return
                else:
                    cl.user_session.set("can_already_start_chat?", True)
                    await ChatProfileHandler.end_init_profile_step()
                    break

                await cl.sleep(1)
        else:
            cl.user_session.set("can_already_start_chat?", True)
            await ChatProfileHandler.end_init_profile_step()

        return params


class ChatHistoryManager:
    @staticmethod
    def init_chat_history(initial_history: Optional[list[dict]] = None):
        if initial_history:
            for message in initial_history:
                ChatHistoryManager.append_to_chat_history(message["content"], message["type"])

        time_info = ChatHistoryManager.get_date_time_info()
        latex_instruct = ChatHistoryManager.instruct_latex_formatting()

        ChatHistoryManager.append_to_chat_history(time_info, "human")
        ChatHistoryManager.append_to_chat_history(latex_instruct, "human")

        # TODO: To be removed
        ChatHistoryManager.clear_chat_history()

    @staticmethod
    def instruct_latex_formatting():
        return (
            "Make sure all math equations are in KaTeX format. "
            "This means you are to encapsulate without failing all the "
            "inline math equations with $ and all the display math equations with $$"
        )

    @staticmethod
    def get_date_time_info():
        today = datetime.now().strftime("%A, %d %B %Y")
        time_now = datetime.now().strftime("%I:%M %p").lower().lstrip("0")

        return f"Today is {today} and the time is {time_now}"

    @staticmethod
    def get_chat_history():
        chat_history = cl.user_session.get("chat_history")

        if chat_history is None:
            chat_history = []
            cl.user_session.set("chat_history", chat_history)

        return chat_history

    @staticmethod
    def clear_chat_history():
        cl.user_session.set("chat_history", [])

    @staticmethod
    def append_to_chat_history(content: str, sender: str):
        chat_history = ChatHistoryManager.get_chat_history()

        chat_history.append(HumanMessage(content=content) if sender == "human" else AIMessage(content=content))

        cl.user_session.set("chat_history", chat_history)

    @staticmethod
    def update_chat_history(question: cl.Message, answer: cl.Message):
        ChatHistoryManager.append_to_chat_history(question.content, "human")
        ChatHistoryManager.append_to_chat_history(answer.content, "ai")

    @staticmethod
    def modify_chat_history(index: int, content: str, sender: str):
        chat_history = ChatHistoryManager.get_chat_history()

        if index < len(chat_history):
            chat_history[index].content = content
            chat_history[index].type = sender

        cl.user_session.set("chat_history", chat_history)


class ChatSettingOptionConfig(BaseModel):
    """Configuration for a chat setting option."""

    id: str
    label: str

    class Config:
        extra = "allow"


class ChatSettingConfig(BaseModel):
    """Configuration for a chat setting."""

    type: str
    options: ChatSettingOptionConfig


class ChatSettingsConfig(RootModel[List[ChatSettingConfig]]):
    """Configuration for multiple chat settings."""

    def input_widgets(self):
        """Generate input widgets based on the settings configuration."""
        return load_input_widgets(self.model_dump())


class ChatProfileConfig(BaseModel):
    """Configuration for a chat profile."""

    name: str
    active: bool
    handler_class: str
    markdown_description: str
    icon: Optional[str] = None
    settings: ChatSettingsConfig
    system_prompt: str
    starters: List[cl.Starter]
    ensure_session_vars: Optional[List[str]] = []
    spontaneous_file_upload: Optional[bool] = False

    class Config:
        extra = "allow"

    def to_cl_chat_profile(self) -> cl.ChatProfile:
        """Convert the configuration to a Chainlit's ChatProfile object."""

        profile_module = self.handler_class.split(".")[0]
        icon_path = self.icon if self.icon else f"public/profiles/{profile_module}/icon.svg"

        # noinspection PyArgumentList
        return cl.ChatProfile(
            name=self.name,
            markdown_description=self.markdown_description,
            icon=icon_path,
            starters=self.starters,
        )
