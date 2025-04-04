from typing import Optional

import chainlit as cl
from pydantic import BaseModel


class HumanInteractionToolSet:
    """An array of human interaction tools."""

    class Action(BaseModel):
        name: str
        value: Optional[str] = None
        label: str

        def to_cl_action(self) -> cl.Action:
            # noinspection PyArgumentList
            return cl.Action(name=self.name, value=self.value or self.name, label=self.label)

    @staticmethod
    async def ask_human(
        query: str,
        parent_id: Optional[str] = None,
        timeout: Optional[int] = 600,
        author: str = "SystemMessage",
    ) -> tuple:
        msg = cl.AskUserMessage(content=query, author=author)
        msg.timeout = timeout
        msg.parent_id = parent_id

        res = await msg.send()

        if res:
            return res["id"], res["output"]
        return None, None

    @staticmethod
    async def ask_human_for_action(
        query: str,
        actions: list[Action],
        default: str = None,
        parent_id: Optional[str] = None,
        timeout: Optional[int] = 600,
        author: str = "SystemMessage",
    ) -> tuple:
        msg = cl.AskActionMessage(
            content=query, actions=[action.to_cl_action() for action in actions], author=author
        )
        msg.timeout = timeout
        msg.parent_id = parent_id

        res = await msg.send()

        if res:
            return res["id"], res["value"]
        return None, default
