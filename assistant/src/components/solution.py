import json
from typing import Any, Union, Callable, Dict, Optional

import chainlit as cl
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from literalai.helper import utc_now
from literalai.step import TrueStepType

from src.logger import log
from src.utils.common import CustomJSONEncoder


# from literalai.observability.step import TrueStepType


class AutoFormattedStep(cl.Step):
    """
    Custom Step class that automatically formats input and output
    based on their data types and structure.
    """

    def __init__(self, name: str, type: TrueStepType, input: Any = None, **kwargs):

        super().__init__(name=name, type=type, **kwargs)

        if input is not None:
            self.input = self._format_content(input)

    def _format_content(self, content: Any) -> str:
        """Format content based on its type"""
        if content is None:
            return ""

        # If already a string, return as is
        if isinstance(content, str):
            return content

        # Handle dictionary and list
        if isinstance(content, (dict, list)):
            try:

                def make_safe(item):
                    if isinstance(item, bytes):
                        return "<STRIPPED_BINARY_DATA>"
                    elif isinstance(item, dict):
                        return {k: make_safe(v) for k, v in item.items()}
                    elif isinstance(item, (list, tuple)):
                        return [make_safe(i) for i in item]
                    return item

                # return f"```json\n{json.dumps(content, cls=CustomJSONEncoder, indent=2)}\n```"
                return f"```json\n{json.dumps(make_safe(content), cls=CustomJSONEncoder, indent=2)}\n```"
            except:
                return str(content)

        # Handle other types
        return str(content)

    @property
    def output(self) -> Optional[str]:
        """Get formatted output"""
        return self._output

    @output.setter
    def output(self, output: Any):
        """Set and format output"""
        self._output = self._format_content(output)


class DualPurposeComponent(BaseTool):
    """
    A component that behaves both as a Tool and RunnableLambda with Chainlit integration.
    """

    func: Union[Callable, Runnable]

    def __init__(
        self, name: str, description: str, func: Union[Callable, Runnable], return_direct: bool = False
    ):
        super().__init__(
            name=name,
            description=description,
            func=func,
            return_direct=return_direct,
        )

        self.func = func.invoke if isinstance(func, Runnable) else func

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """
        Async run implementation with Chainlit Step
        """

        input_data = (
            {"args": args, **kwargs}
            if args and kwargs
            else ((args[0] if len(args) == 1 else args) if args else kwargs)
        )

        async with AutoFormattedStep(name=self.name, type="tool", input=input_data) as step:
            try:
                # Refresh the step before begin
                step.output = {"result": "Tool execution in progress ..."}
                await step.update()

                # Execute the function
                result = self.func(input_data)

                # Convert each output to a dictionary as that is the default
                # expected output type of each Lambda in a RunnableSequence
                if not isinstance(result, dict):
                    result = {"result": result}

                # Set step output
                step.output = result

                return result

            except Exception as ex:
                step.output = f"Error: {str(ex)}"
                log.error(f"Error occurred: {ex}", exc_info=ex)
                raise ex

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        input_data = (
            {"args": args, **kwargs}
            if args and kwargs
            else ((args[0] if len(args) == 1 else args) if args else kwargs)
        )

        return self.func(input_data)

    def invoke(self, input: Union[str, dict], config=None, **kwargs) -> Any:
        return self.func(input)


class ToolChain:
    """
    ToolChain with Chainlit integration
    """

    def __init__(
        self,
        name: str,
        components: Union[BaseTool, list[BaseTool], dict[str, Union[BaseTool, list[BaseTool]]]],
    ):
        self.name: str = name.replace(" ", "")
        self.components: Union[list[BaseTool], dict[str, Union[BaseTool, list[BaseTool]]]] = components

    async def arun(self, initial_input: Any) -> Any:
        """Async run method with Chainlit Steps"""

        async with AutoFormattedStep(
            name=self.name, type="tool", input=initial_input, show_input=True
        ) as parent_step:
            # Refresh the step before begin
            parent_step.output = {"result": "Tool Chain execution in progress ..."}
            await parent_step.update()

            try:
                # Initialize the current output
                current_output = initial_input

                # Run the tool chain
                if isinstance(self.components, dict):
                    # Run each tool chain in the dictionary
                    dict_chain_outputs = {}
                    for key, component_list in self.components.items():
                        dict_chain_outputs[key] = await ToolChain(
                            name=key,
                            components=component_list,
                        ).arun(current_output)
                        current_output = dict_chain_outputs[key]
                elif not isinstance(self.components, list):
                    self.components = [self.components]

                if isinstance(self.components, list):
                    # Iterate over each component
                    for component in self.components:
                        if isinstance(component, DualPurposeComponent):
                            current_output = await component.arun(current_output)
                        else:
                            async with AutoFormattedStep(
                                name=component.name, type="tool", input=current_output, show_input=True
                            ) as step:
                                # Refresh the step before begin
                                step.output = {"result": "Tool execution in progress ..."}
                                await step.update()

                                current_output = component.run(current_output)
                                step.output = current_output

                # Set final output for parent step
                parent_step.output = current_output

                return current_output

            except Exception as ex:
                parent_step.output = f"Error: {str(ex)}"
                log.error(f"Error occurred: {ex}", exc_info=ex)
                raise ex

    def run(self, initial_input: Any) -> Any:
        """Synchronous run method"""

        current_output = initial_input
        for component in self.components:
            if isinstance(component, DualPurposeComponent):
                current_output = component.invoke(current_output)
            else:
                current_output = component.run(current_output)
        return current_output


class ToolWithSources:
    """
    A class to manage tool execution steps and associated sources.

    This class provides functionality to initialize tools, handle tool events,
    and update sources based on tool execution outputs.
    """

    def __init__(self) -> None:
        # noinspection PyUnresolvedReferences
        """
        Initialize the ToolWithSources instance.

        Attributes:
            steps (Dict[str, cl.Step]): A dictionary to store tool execution steps.
            sources (Dict[str, cl.Text]): A dictionary to store sources associated with tool executions.
        """
        self.steps: Dict[str, cl.Step] = {}
        self.sources: Dict[str, cl.Text] = {}

    @staticmethod
    def init_tool(*args, **kwargs):
        """
        Initialize a tool with given arguments and keyword arguments.

        This method is a placeholder and should be implemented in subclasses.
        """
        pass

    @staticmethod
    def get_tool(*args, **kwargs):
        """
        Get the tool instance.

        This method is a placeholder and should be implemented in subclasses.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def update_sources(self, event_output: list[dict]):
        """
        Update sources based on the event output.

        This method is a placeholder and should be implemented in subclasses.

        Args:
            event_output (list[dict]): A list of dictionaries containing event output data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def handle_tool_events(self, event):
        """
        Handle tool events based on their type.

        Args:
            event (dict): A dictionary containing event data.
        """
        event_type = event["event"]
        if event_type == "on_tool_start":
            await self.on_tool_call_created(event)
        elif event_type == "on_tool_end":
            await self.on_tool_call_done(event)

    async def on_tool_call_created(self, event):
        """
        Handle the creation of a tool call.

        This method creates a new Step instance and sends it.

        Args:
            event (dict): A dictionary containing event data for tool call creation.
        """
        step = AutoFormattedStep(
            name=event["name"],
            type="tool",
            parent_id=cl.context.current_run.id,
        )

        step.input = event["data"].get("input")
        step.start = utc_now()

        await step.send()

        self.steps[event["run_id"]] = step

    async def on_tool_call_done(self, event):
        """
        Handle the completion of a tool call.

        This method updates the Step instance with the output and end time,
        and processes the event output.

        Args:
            event (dict): A dictionary containing event data for tool call completion.
        """
        step = self.steps.get(event["run_id"])

        if step:
            step.end = utc_now()

            event_output = event["data"].get("output")
            step.output = event_output

            if event_output:
                if isinstance(event_output, str):
                    # noinspection PyBroadException
                    try:
                        event_output = json.loads(event_output)
                    except Exception as ex:
                        log.error(f"Error on converting step.output to json: {ex}")

                if isinstance(event_output, list) and isinstance(event_output[0], dict):
                    self.update_sources(event_output)

                await step.update()
