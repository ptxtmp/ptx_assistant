import chainlit as cl
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.components.chat import ChatProfileHandler, ChatHistoryManager
from src.utils.llms import create_hugging_face_model


class GeneralAssistant(ChatProfileHandler):
    """
    A general-purpose chat profile that provides basic conversational capabilities.
    This profile is designed for general queries and discussions without specific domain focus.
    """

    @staticmethod
    async def start(config, settings: cl.ChatSettings):
        """Initialize the general chat session."""

        # Get the settings
        settings: dict = await settings.send()

        profile_config = config.chat_profiles.general

        # Initialize chat history
        ChatHistoryManager.init_chat_history()

        # Initialize the language model
        llm, _ = create_hugging_face_model(config)

        # Create the prompt template - using a format that ChatHuggingFace will understand
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=profile_config.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create the chain with streaming support
        chain = prompt | llm | StrOutputParser()

        # Set the model
        cl.user_session.set("model", chain)

    @staticmethod
    async def main(config, message: cl.Message):
        """Handle incoming messages in the general chat."""
        try:
            # Get the model
            model = cl.user_session.get("model")

            # Get the user's message
            user_message = message.content

            # Get chat history
            chat_history = ChatHistoryManager.get_chat_history()

            # Create a message placeholder for streaming
            msg = cl.Message(content="", author=config.app_name)
            await msg.send()

            # Stream the response
            stream = False
            full_message = ""
            human_prefix = "Human:"
            ai_prefix = "Assistant:"
            async for chunk in model.astream({
                "chat_history": chat_history,
                "input": f"{human_prefix} {user_message}"
            }):
                if stream:
                    await msg.stream_token(chunk)
                else:
                    full_message += chunk
                    if ai_prefix in full_message:
                        full_message = full_message.split(ai_prefix)[1]
                        await msg.stream_token(full_message)
                        stream = True

            # Update the message
            await msg.update()

            # Update chat history with the complete message
            ChatHistoryManager.update_chat_history(message, msg)

        except Exception as e:
            # Log the error and inform the user
            error_msg = f"An error occurred: {str(e)}"
            await cl.Message(content=error_msg, author="System").send()
            raise

    @staticmethod
    async def settings_update(config, settings: dict):
        """Handle settings updates for the general chat."""
        # Add any settings update logic here if needed
        pass

    @staticmethod
    async def end(config):
        """Clean up when the chat session ends."""
        ChatHistoryManager.clear_chat_history()
