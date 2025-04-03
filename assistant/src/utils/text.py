from io import BytesIO

from docx import Document
from pypdf import PdfReader

from src.logger import log


async def chunk_text_async(splitter, text):
    try:
        return splitter.split_text(text)
    except Exception as ex:
        log.error(str(ex), exc_info=ex)
        return ""


def read_text_from_file(file_path: str):
    # Read file as bytes
    with open(file_path, "rb") as uploaded_file:
        file_contents = uploaded_file.read()
    log.info("[%d] bytes were read from %s", len(file_contents), file_path)

    # Convert bytes to BytesIO
    bytes_io = BytesIO(file_contents)

    text = ""
    extension = file_path.split(".")[-1]
    if extension == "txt":
        text = file_contents.decode()
    elif extension == "pdf":
        text = pdf_text_extractor(bytes_io)
    elif extension == "docx":
        text = docx_text_extractor(bytes_io)
    else:
        log.error("Not a valid file format!")

    return text


def pdf_text_extractor(bytes_io: BytesIO):
    # Initialize the PdfReader
    reader = PdfReader(bytes_io)

    # Extract text from Pdf file
    page_texts = []
    for i in range(len(reader.pages)):
        page_text = reader.pages[i].extract_text()
        page_texts.append(page_text)

    return "\n".join(page_texts)


def docx_text_extractor(bytes_io: BytesIO):
    # Initialize the docx Document
    doc = Document(bytes_io)

    # Extract text from document
    paragraph_list = []
    for paragraph in doc.paragraphs:
        paragraph_list.append(paragraph.text)

    return "\n".join(paragraph_list)
