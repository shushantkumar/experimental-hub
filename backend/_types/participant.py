"""TODO Document"""

from typing import Any, TypedDict

from _types.size import SizeDict
from _types.position import PositionDict


class __ParticipantDictOptionalKeys(TypedDict, total=False):
    """TODO Document"""
    id: str


class ParticipantDict(__ParticipantDictOptionalKeys):
    """TODO Document"""
    first_name: str
    last_name: str
    muted: bool
    filters: list[Any]
    position: PositionDict
    size: SizeDict
