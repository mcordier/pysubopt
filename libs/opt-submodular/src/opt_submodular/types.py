from typing import Generic, TypeVar, List

from pydantic import BaseModel

T = TypeVar("T")

class Result(BaseModel, Generic[T]):
    opt_subset: List[T]
    fun_value: float
    cost_fun_value: float
    delta_time: float
