import json
import os
from functools import wraps
from typing import Dict, Optional

import chainlit as cl
from dotenv import load_dotenv

from config import AppConfig
from src.components.chat import ChatProfileHandler
from src.logger import log
from src.utils.common import dict_to_md_table

if os.path.exists(".env"):
    load_dotenv(override=True)

# Load App configs
config = AppConfig.load_from_yaml("chat.config.yaml")


# noinspection PyUnusedLocal
@cl.set_chat_profiles
async def set_chat_profiles(current_user: cl.User):
    return config.get_cl_chat_profile_list()


# @cl.oauth_callback
# async def oauth_callback(
#     provider_id: str,
#     token: str,
#     raw_user_data: Dict[str, str],
#     default_user: cl.User,
# ) -> Optional[cl.User]:
#
#     log.debug(
#         f"\nprovider_id={provider_id},\ntoken={token},\nraw_user_data={raw_user_data},\ndefault_user={default_user}"
#     )
#
#     if provider_id == "google":
#         return default_user
#
#     # Send success message
#     await cl.Message(content="Successfully authenticated! Refreshing page...", author="System").send()
#
#     # Wait briefly for the message to be shown
#     await cl.sleep(3)
#
#     # Trigger page refresh
#     # noinspection PyArgumentList
#     await cl.Action(name="refresh", value="reload", script="window.location.reload();").send()
#
#     return None


#######################################
# Error handleable chainlit functions #
#######################################


def error_handler(func):
    """
    A decorator that wraps the chat start function with error handling.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)

        except Exception as ex:
            import traceback
            from datetime import datetime

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log.error(f"Error occurred at {current_time}: {ex}", exc_info=ex)

            # noinspection PyArgumentList
            elements = [
                cl.Text(
                    content=f"```bash\n{traceback.format_exc()}\n```", name="StackTrace", display="side"
                )
            ]

            msg = cl.Message(
                content=f"A system error has occurred at `{current_time}`.\n"
                f"> **Error message:**\n```bash\n{ex}\n```\n"
                f"If you wish to report this error, please share the screenshot of above error message.\n"
                f"Click here for additional StackTrace.",
                elements=elements,
                author="SystemMessage",
            )
            await msg.send()

    return wrapper


async def notify_settings_update(settings: dict):
    keys = set(list(settings.keys()))
    if "SystemPrompt" in keys:
        keys = keys - {"SystemPrompt"}

    # noinspection PyArgumentList
    elements = [
        cl.Text(
            content=(
                f"Settings Table\n---\n"
                + f"{dict_to_md_table([{k: settings[k] for k in keys}], transpose=True)}\n\n"
                + (
                    f"System Prompt\n---\n```yaml\n{settings['SystemPrompt']}\n```\n\n"
                    if "SystemPrompt" in settings.keys()
                    else ""
                )
                + f"Settings JSON\n---\n```json\n{json.dumps(settings, indent=2)}\n```"
            ),
            name="Settings",
            display="side",
        )
    ]

    content = f"New Settings has been applied!"
    if "Model" in settings and settings["Model"] in ("o1", "o1-mini"):
        content = (
            f"{content}\n\n_o1 class models does not support streaming the response,"
            f" which means you may have to wait a bit before you see the answer to your query._"
        )

    msg = cl.Message(
        content=content,
        elements=elements,
        author="SystemMessage",
    )
    await msg.send()


@cl.on_settings_update
@error_handler
async def settings_update(settings: dict):

    log.debug(f"Settings Updated: {settings}")
    selected_chat_profile = cl.user_session.get("chat_profile")

    if selected_chat_profile:

        updated_settings = await config.get_chat_handler_class(
            chat_profile=selected_chat_profile, key="name"
        ).settings_update(config, settings)

        if updated_settings:

            settings = await config.sync_chat_profile_settings(
                chat_profile=selected_chat_profile, settings=updated_settings
            )

        # Notification: "Your Settings has been applied!"
        await notify_settings_update(settings)

    else:

        log.error("cl.user_session.get('chat_profile') returns either None or empty")


@cl.on_chat_start
@error_handler
async def start():

    selected_chat_profile = cl.user_session.get("chat_profile")

    if selected_chat_profile:

        # Initialize the profile
        await ChatProfileHandler.start_init_profile_step()

        # Get the settings
        settings = config.get_chat_profile_settings(chat_profile=selected_chat_profile, key="name",)

        # Start the profile
        await config.get_chat_handler_class(chat_profile=selected_chat_profile, key="name",).start(
            config, settings
        )

        # End the initialization step
        await ChatProfileHandler.end_init_profile_step()

    else:

        log.error("cl.user_session.get('chat_profile') returns either None or empty")


async def handle_spontaneous_file_upload(message: cl.Message, selected_chat_profile: str):

    if message.elements and any(
        [isinstance(element, (cl.element.File, cl.element.Image)) for element in message.elements]
    ):
        await cl.Message(
            f"It seems you have done a spontaneous file upload. This feature is not "
            f"supported in the `{selected_chat_profile}` profile. If you would like to "
            "interact with a document, please consider using the document upload feature "
            "provided in the **Chat with Doc** profile. It also supports spontaneous file upload.",
            author="SystemMessage",
        ).send()
        return True

    return False


@cl.on_message
@error_handler
async def main(message: cl.Message):

    selected_chat_profile = cl.user_session.get("chat_profile")

    if selected_chat_profile:

        # Get the profile config
        profile_config = config.get_chat_profile(chat_profile=selected_chat_profile, key="name",)

        # Initialize the profile
        if profile_config.ensure_session_vars:

            # Check if the profile can already start the chat
            start_chat = cl.user_session.get("can_already_start_chat?")
            if not start_chat:

                # Start the initialization step
                await ChatProfileHandler.start_init_profile_step()

                # Ensure the user session values
                params = await ChatProfileHandler.ensure_user_session_values(
                    config, profile_config.ensure_session_vars
                )
                if not params:
                    return

            # End the initialization step
            await ChatProfileHandler.end_init_profile_step()

            # Set the flag to indicate the profile can already start the chat
            cl.user_session.set("can_already_start_chat?", True)

        # Handle spontaneous file upload
        if not profile_config.spontaneous_file_upload:
            if await handle_spontaneous_file_upload(message, selected_chat_profile):
                return

        # Start the chat
        await config.get_chat_handler_class(chat_profile=profile_config).main(config, message)

    else:

        log.error("cl.user_session.get('chat_profile') returns either None or empty")


@cl.on_stop
@error_handler
async def stop():

    selected_chat_profile = cl.user_session.get("chat_profile")

    if selected_chat_profile:

        await config.get_chat_handler_class(
            chat_profile=selected_chat_profile,
            key="name",
        ).stop(config)

    else:

        log.error("cl.user_session.get('chat_profile') returns either None or empty")


@cl.on_chat_end
@error_handler
async def end():
    selected_chat_profile = cl.user_session.get("chat_profile")

    if selected_chat_profile:

        await config.get_chat_handler_class(
            chat_profile=selected_chat_profile,
            key="name",
        ).end(config)

    else:

        log.error("cl.user_session.get('chat_profile') returns either None or empty")


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
