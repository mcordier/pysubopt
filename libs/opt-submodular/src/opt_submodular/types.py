"""Module defining the different types used in the package.
"""
from typing import Generic, List, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Result(BaseModel, Generic[T]):

    """Result object returned after an optimization
    process.
    """

    opt_subset: List[T]
    fun_value: float
    cost_fun_value: float
    delta_time: float
