import json
import mimetypes
import os
from typing import List, cast, Any, Optional, Union

import chainlit as cl
from chainlit.element import ElementBased, Element
from chainlit.types import AskFileResponse
from config import AppConfig
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from src.components.chat import ChatProfileHandler, ChatHistoryManager
from src.components.solution import ToolWithSources
from src.logger import log
from src.utils.common import format_list_for_msg
from src.utils.llms import create_chat_openai_model
from src.utils.text import read_text_from_file, chunk_text_async

# Define the local cache directory for sentence transformer models
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models",
                               "sentence_transformers")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


class FileElement(Element):
    """A chainlit Element class for handling files with path and mime attributes."""

    type: str = "file"

    def __init__(self, filepath: str):
        self.path = filepath
        self.name = os.path.basename(filepath)
        self.mime = mimetypes.guess_type(filepath)[0]

        super().__init__(
            type=self.type,
            name=self.name,
            display="side",
            mime=self.mime,
            path=self.path
        )


class FileUploadHandler:

    def __init__(
            self,
            file_upload_config: AppConfig,
            vectorization_config: AppConfig,
            debug: bool = False,
    ):

        self.debug: bool = debug

        self.msg = cl.Message(content="", author="SystemMessage")
        self.files: list[Union[AskFileResponse, ElementBased]] = []
        self.text_splitter = None

        self.file_upload_conf: AppConfig = file_upload_config
        self.vectorize_conf: AppConfig = vectorization_config

    async def run(self):

        if not self.files:
            await self.prompt_file_upload()

        await self.msg.send()

        processed_files, processed_data = await self.process_uploaded_files()

        file_names = [f.name for f in processed_files]

        return self.msg, file_names, processed_data

    async def prompt_file_upload(self):

        user = cl.user_session.get("user")

        file_formats = format_list_for_msg(self.file_upload_conf.allowed_file_formats)

        # Prompt user to upload files and wait for the upload
        while not self.files:
            self.files = await cl.AskFileMessage(
                # f"Hi Max! Please upload up to 3 `.pdf` files up to **5** MB to begin.",
                content=self.file_upload_conf.prompt.format(
                    user=user.identifier if user and user.identifier else "User",
                    max_files=self.file_upload_conf.max_files,
                    file_formats=file_formats,
                    max_size=self.file_upload_conf.max_size_mb,
                ),
                accept=self.file_upload_conf.accepted_mime_types,
                max_size_mb=self.file_upload_conf.max_size_mb,
                max_files=self.file_upload_conf.max_files,
                timeout=86400,
                raise_on_timeout=False,
            ).send()

    def init_text_splitter(self):
        if not self.text_splitter:
            self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=self.vectorize_conf.text_splitter.chunk_size,
                chunk_overlap=self.vectorize_conf.text_splitter.chunk_overlap,
            )

    async def update_progress_message(self, process: str, files: list[AskFileResponse], trail: bool = True):
        # Notify the files that are going to be processed
        files_names = format_list_for_msg([file.name for file in files], inclusive=True)
        content = f"{process} {files_names}{'...' if trail else '.'}"
        log.info(content)

        self.msg.content = content
        await self.msg.update()

        await cl.sleep(1)

    async def handle_invalid_files(self):
        accepted_mime_types = self.file_upload_conf.accepted_mime_types

        # Check for invalid files
        invalid_files = [
            file
            for file in self.files
            if not isinstance(file, AskFileResponse) and file.mime and file.mime not in accepted_mime_types
        ]
        if invalid_files:
            file_formats = format_list_for_msg(self.file_upload_conf.allowed_file_formats)
            file_names = format_list_for_msg(
                [file.name for file in invalid_files],
                inclusive=True,
                singular_appends=(" ", " is"),
                plural_appends=("s ", " are"),
            )

            await cl.Message(
                self.file_upload_conf.wrong_file_type_message.format(
                    file_name=file_names,
                    max_files=self.file_upload_conf.max_files,
                    file_formats=file_formats,
                    max_size=self.file_upload_conf.max_size_mb,
                ),
                author="SystemMessage",
            ).send()

        return [
            file
            for file in self.files
            if isinstance(file, AskFileResponse) or (file.mime and file.mime in accepted_mime_types)
        ]

    @cl.step(type="tool")
    async def process_uploaded_files(self):

        # Update progress message
        await self.update_progress_message(process="Processing", files=self.files)

        output_data = None
        processed_files = []

        # Check for valid files
        valid_files = await self.handle_invalid_files()
        if valid_files:
            # Initialize file splitter
            self.init_text_splitter()

            # Read and split texts
            texts = {}
            for file in valid_files:

                # Update progress message
                await self.update_progress_message(process="Reading", files=[file])

                # Read file text
                text = read_text_from_file(file.path).strip()
                if self.debug:
                    log.debug("[%s] characters read from file %s", len(text), file.path)

                if not text:
                    # Update progress message
                    await self.update_progress_message(
                        process="No text could be extracted from", files=[file], trail=False
                    )
                    continue

                # Update progress message
                await self.update_progress_message(process="Chunking", files=[file])

                # Split file text
                texts[file.name] = await chunk_text_async(self.text_splitter, text)

                # Append file name to each text chunk
                texts[file.name] = [f"Filename: {file.name}\n---\n\n{text}" for text in texts[file.name]]

                # Add file to processed files
                processed_files.append(file)

            # Prepare text metadata
            output_data = [
                {"source_name": f"{file_name}-chunk-{i}", "content": text}
                for file_name, texts in texts.items()
                for i, text in enumerate(texts)
            ]

            if self.debug:
                log.debug(f"Processed Data:\n{json.dumps(output_data, indent=2)}")

            # Update progress message
            if output_data:
                await self.update_progress_message(
                    process="Processing complete for", files=processed_files, trail=False
                )

                if len(valid_files) != len(processed_files):
                    unprocessed_files = format_list_for_msg(
                        [file.name for file in valid_files if file not in processed_files], inclusive=True
                    )
                    await cl.Message(
                        f"No text data could be extracted from {unprocessed_files}.",
                        author="SystemMessage",
                    ).send()
            else:
                await self.update_progress_message(
                    process="No text data could be extracted from", files=self.files, trail=False
                )

        return processed_files, output_data


class DocumentRetrieverTool(ToolWithSources):
    embedder: Optional[SentenceTransformerEmbeddings] = None

    @staticmethod
    def set_embedder(config: AppConfig, vectorization_config: AppConfig):
        # Initialize SentenceTransformer with a high-quality model and local cache
        DocumentRetrieverTool.embedder = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2",  # A good balance of quality and speed
            cache_folder=MODEL_CACHE_DIR,  # Use local cache directory
            model_kwargs={
                "device": "cpu",  # Use GPU if available
            },
            encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search
        )

    @staticmethod
    def init_tool(config: AppConfig, vectorization_config: AppConfig, processed_files_data: List[dict]):

        # Set the embedder
        DocumentRetrieverTool.set_embedder(config, vectorization_config)

        # Vectorize the chunked data
        DocumentRetrieverTool.vectorize_chunked_data(processed_files_data)

    @staticmethod
    def get_tool():
        return DocumentRetrieverTool.chat_with_document

    @staticmethod
    @cl.step(type="tool")
    def vectorize_chunked_data(processed_files_data: List[dict]):

        all_texts = [data["content"] for data in processed_files_data]
        text_meta_datas = [{"source_name": data["source_name"]} for data in processed_files_data]

        if all_texts and text_meta_datas:
            # noinspection PyArgumentList
            # Vectorize all texts: "This vector DB is in-memory, no user data is stored on disk"
            chroma_vector_db = Chroma.from_texts(
                all_texts,
                DocumentRetrieverTool.embedder,
                metadatas=text_meta_datas,
            )

            cl.user_session.set("doc_vector_db", chroma_vector_db.as_retriever())

            return "`All chunked text data is vectorized successfully!`"

        else:
            return "`No chunked text data could be vectorized!`"

    @staticmethod
    @tool
    def chat_with_document(query: str):
        """Retrieve relevant information from the uploaded document based on the given query.
        This function performs a semantic search on the vectorized content of the uploaded document.
        It returns the most relevant text passages that match the query, allowing the LLM to provide
        accurate answers based solely on the information contained within the uploaded document.
        The returned results can include any type of information present in the document, such as
        facts, figures, explanations, or specific details related to the query.
        """

        processed_docs = []
        retriever = cl.user_session.get("doc_vector_db")

        if retriever:
            docs: list[Document] = retriever.invoke(input=query)
            for idx, doc in enumerate(docs):
                proc_doc = dict()

                proc_doc["source_name"] = doc.metadata.get("source_name")
                proc_doc["content"] = doc.page_content

                processed_docs.append(proc_doc)

        return processed_docs

    def update_sources(self, event_output: list[dict]):
        sources = {
            res["source_name"]: res.get("content")
            for i, res in enumerate(event_output)
            if res.get("content")
        }

        self.sources = {**self.sources, **sources}


class PTXAssistant(ChatProfileHandler):

    @staticmethod
    async def start(config: AppConfig, settings: cl.ChatSettings):

        settings: dict[str, Any] = await settings.send()

        profile_config = config.chat_profiles.ptx_assistant

        file_upload_conf = profile_config.file_upload
        file_upload_conf["prompt"] = profile_config.file_upload.prompt

        file_upload_handler = FileUploadHandler(
            file_upload_config=file_upload_conf,
            vectorization_config=profile_config.vectorization,
            debug=config.debug,
        )

        if profile_config.file_upload.preload_files:
            file_upload_handler.files.extend([FileElement(f) for f in profile_config.file_upload.preload_files])

        msg, files, processed_files_data = await file_upload_handler.run()

        DocumentRetrieverTool.init_tool(config, profile_config.vectorization, processed_files_data)

        # Update the processing message
        if not processed_files_data:
            files_names = ", ".join([f"`{f}`" for f in files])
            content = f"{files_names} could not be processed. Please try again!"
            log.info(content)

            msg.content = content
            await msg.update()

        # Initialize LLM model
        llm, _ = create_chat_openai_model(config=config)

        # Define the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", profile_config.system_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", profile_config.human_prompt),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Define the toolkit
        toolkit = [DocumentRetrieverTool.get_tool()]

        # Create the agent
        # noinspection PyTypeChecker
        agent = create_tool_calling_agent(llm, toolkit, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=config.debug)

        # Set the agent
        cl.user_session.set("agent", agent_executor)

        # Update the processing message
        if processed_files_data:
            files_names = format_list_for_msg(
                files, inclusive=True, singular_appends=(" ", " is"), plural_appends=("s ", " are")
            )
            content = f"File{files_names} processed. You can now ask questions."
            log.info(content)

            msg.content = content
            await msg.update()

        # Update the settings
        await PTXAssistant.settings_update(config, settings)

        # Set the chat history
        ChatHistoryManager.init_chat_history()

    @staticmethod
    async def settings_update(config: AppConfig, settings: dict):

        profile_config = config.chat_profiles.ptx_assistant

        cl.user_session.set(
            "limit_answer",
            (
                profile_config.conditional_prompt.limit_answer_to_doc
                if settings["limit_answer"]
                else profile_config.conditional_prompt.answer_without_limits
            ),
        )

    @staticmethod
    async def handle_spontaneous_file_upload(
            message: cl.Message, profile_config: AppConfig, debug: bool = False
    ):

        if message.elements and any(
                [isinstance(element, (cl.element.File, cl.element.Image)) for element in message.elements]
        ):

            file_upload_conf = profile_config.file_upload
            file_upload_conf["prompt"] = profile_config.file_upload.prompt

            # Process the uploaded files
            file_upload_handler = FileUploadHandler(
                file_upload_config=file_upload_conf,
                vectorization_config=profile_config.vectorization,
                debug=debug,
            )
            file_upload_handler.files.extend(message.elements)
            msg, files, processed_files_data = await file_upload_handler.run()

            # Vectorize the chunked data
            DocumentRetrieverTool.vectorize_chunked_data(processed_files_data)

            # Update the processing message
            if processed_files_data:
                files_names = format_list_for_msg(
                    files, inclusive=True, singular_appends=(" ", " is"), plural_appends=("s ", " are")
                )
                content = f"File{files_names} processed. You can now ask questions."
                log.info(content)

                msg.content = content
                await msg.update()

    @staticmethod
    async def main(config: AppConfig, question: cl.Message):

        # Handle spontaneous file upload
        await PTXAssistant.handle_spontaneous_file_upload(question, config.chat_profiles.ptx_assistant)

        # Get the chat history
        chat_history = cl.user_session.get("chat_history")

        # Get the agent
        runnable = cast(Runnable, cl.user_session.get("agent"))

        # Get the limit answer
        limit_current_answer = cl.user_session.get("limit_answer")

        elements = []
        document_retriever = DocumentRetrieverTool()

        answer = cl.Message(content="", author=config.app_name)
        async for event in runnable.astream_events(
            {
                "question": question.content,
                "limit_answer_to_doc": limit_current_answer,
                "chat_history": chat_history,
            },
            version="v2",
        ):
            # type not enforced right now to avoid overhead during streaming
            # from langchain_core.runnables.schema import StreamEvent
            # event_casted = cast(StreamEvent, event)
            # Improved version should only pass custom EventHandler class into stream
            # https://github.com/Chainlit/cookbook/blob/90d540b72ebe7bbcda8d2fb8b41e8cdd23571da6/openai-data-analyst/app.py#L211
            # Key Open Decision: Custom Agent Layer in Langchain vs. "Assistant" from AzureOpenAI and async_azure_openai_client.beta.threads.runs.stream()

            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    await answer.stream_token(content)
            elif kind.startswith("on_tool_"):
                await document_retriever.handle_tool_events(event)

        # Update chat history before sending the source information
        ChatHistoryManager.update_chat_history(question, answer)

        sources = document_retriever.sources
        if sources:
            for url, content in sources.items():
                # noinspection PyArgumentList
                elements.append(cl.Text(content=content, name=url, display="side"))

            source_names = "\n".join([f" - {source}" for source in sorted(sources.keys())])
            await answer.stream_token(f"\n\n**Sources:**\n{source_names}")

        answer.elements = elements
        await answer.send()

    @staticmethod
    async def end(config: AppConfig):
        cl.user_session.delete("agent")
        cl.user_session.delete("chat_history")
        cl.user_session.delete("limit_answer")
        cl.user_session.delete("doc_vector_db")
        log.info("Chat with document session ended")
