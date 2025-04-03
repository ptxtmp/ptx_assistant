import dataclasses
import datetime
import importlib
import re
import sys
from decimal import Decimal
from json import JSONEncoder
from pathlib import Path
from typing import Optional, Any
from uuid import UUID

from src.logger import log


def load_class(module_path: str):
    """
    Helper function to load class which was defined as a string in a config

    Args:
        module_path (str): A path to the class that should be imported in the following form:
            `path_to_module_with_class.class_name`
    """
    module = ".".join(module_path.split(".")[:-1])
    class_in_module = module_path.split(".")[-1]

    log.debug(f"Module: {module}; Class: {class_in_module}")

    # noinspection PyBroadException
    try:
        if module.startswith("."):
            module = importlib.import_module(module, package=__name__)
        else:
            module = importlib.import_module(module)
        log.debug(f"Imported Module: {module}")

        imported_class = getattr(module, class_in_module)
        log.debug(f"Imported Class: {imported_class}")
    except Exception as ex:
        log.warning(f"'load_class' could not find class {class_in_module} in module {module}")
        log.warning(f"Original Exception: {str(ex)}")
        # Look for class in current script
        module = sys.modules[__name__]
        log.debug(f"Imported Module: {module}")

        imported_class = getattr(module, class_in_module)
        log.debug(f"Imported Class: {imported_class}")

    return imported_class


def flatten_list(list_obj: list):
    flattened_list = []
    for item in list_obj:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def dict_to_md_table(
    data: list[dict], headers: Optional[list] = None, sort_headers: bool = False, transpose: bool = False
):
    if not data:
        return ""

    sort_headers = False if headers else sort_headers

    headers = data[0].keys() if not headers else headers
    headers = sorted(headers) if sort_headers else headers

    if transpose:
        # Transposed table
        markdown_table = (
            "| Key | "
            + " | ".join((f"Item {i+1}" for i in range(len(data))) if len(data) > 1 else ["Value"])
            + " |\n"
        )
        markdown_table += "| --- | " + " | ".join(["---" for _ in data]) + " |\n"

        for header in headers:
            markdown_table += (
                f"| {header} | "
                + " | ".join(str(row.get(header, "")).replace("\n", "  ") for row in data)
                + " |\n"
            )
    else:
        # Regular table for multiple items
        markdown_table = "| " + " | ".join(headers) + " |\n"
        markdown_table += "| " + " | ".join(["---" for _ in headers]) + " |\n"

        for row in data:
            markdown_table += (
                "| " + " | ".join(str(row.get(header, "")).replace("\n", "  ") for header in headers) + " |\n"
            )

    return markdown_table


def format_list_for_msg(
    items: list[str],
    sep: str = ",",
    quote: bool = True,
    inclusive: bool = False,
    singular_appends: tuple = None,
    plural_appends: tuple = None,
):

    sep = "," if not sep else sep
    quote = "`" if quote else ""
    inclusive = "and" if inclusive else "or"

    plural_appends = ("", "") if not plural_appends else plural_appends
    singular_appends = ("", "") if not singular_appends else singular_appends

    output = [f"{quote}{ff}{quote}" for ff in items[:-1]]
    output = (
        f"{plural_appends[0]}{f'{sep} '.join(output)} {inclusive} {quote}{items[-1]}{quote}{plural_appends[1]}"
        if len(items) > 1
        else f"{singular_appends[0]}{quote}{items[0]}{quote}{singular_appends[1]}"
    )
    return output


def convert_latex_content(content: str):

    inline_latex_pattern = r"\\\((.+)\\\)"
    content = re.sub(inline_latex_pattern, r"$\1$", content)

    display_latex_pattern = r"\\\[(.+)\\\]"
    content = re.sub(display_latex_pattern, r"$$\1$$", content)

    return content


class DotDict(dict):
    """
    A dictionary subclass that allows dot notation access to dictionary items.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    @classmethod
    def from_dict(cls, data: dict):
        if isinstance(data, dict):
            return cls({key: cls.from_dict(value) for key, value in data.items()})
        elif isinstance(data, list):
            return [cls.from_dict(item) for item in data]
        else:
            return data

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:

        # Handle datetime objects
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()

        # Handle Decimal
        if isinstance(obj, Decimal):
            return str(obj)

        # Handle UUID
        if isinstance(obj, UUID):
            return str(obj)

        # Handle bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8')

        # Handle set
        if isinstance(obj, set):
            return list(obj)

        # Handle PathLib objects
        if isinstance(obj, Path):
            return str(obj)

        # Handle dataclasses
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        # Handle objects with __dict__ attribute
        if hasattr(obj, '__dict__'):
            return obj.__dict__

        # Handle objects with to_json method
        if hasattr(obj, 'to_json'):
            return obj.to_json()

        # Let the base class handle the remaining
        return JSONEncoder.default(self, obj)
