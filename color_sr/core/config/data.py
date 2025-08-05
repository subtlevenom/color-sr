from pydantic import BaseModel
from enum import Enum


class DataType(str, Enum):
    img = 'img' 


class DataPathes(BaseModel):
    source: str
    target: str


class Data(BaseModel):
    type: DataType = DataType.img
    train: DataPathes
    val: DataPathes
    test: DataPathes
    folds: int = 1