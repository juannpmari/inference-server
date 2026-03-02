from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StoreBlockRequest(_message.Message):
    __slots__ = ("block_id", "block_hash", "data", "model_id", "layer_name")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_HASH_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_NAME_FIELD_NUMBER: _ClassVar[int]
    block_id: int
    block_hash: str
    data: bytes
    model_id: str
    layer_name: str
    def __init__(self, block_id: _Optional[int] = ..., block_hash: _Optional[str] = ..., data: _Optional[bytes] = ..., model_id: _Optional[str] = ..., layer_name: _Optional[str] = ...) -> None: ...

class StoreBlockResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class LoadBlockRequest(_message.Message):
    __slots__ = ("block_id", "layer_name")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    LAYER_NAME_FIELD_NUMBER: _ClassVar[int]
    block_id: int
    layer_name: str
    def __init__(self, block_id: _Optional[int] = ..., layer_name: _Optional[str] = ...) -> None: ...

class LoadBlockResponse(_message.Message):
    __slots__ = ("success", "data", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    data: bytes
    message: str
    def __init__(self, success: bool = ..., data: _Optional[bytes] = ..., message: _Optional[str] = ...) -> None: ...

class GetFreeBlocksRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetFreeBlocksResponse(_message.Message):
    __slots__ = ("num_free_blocks",)
    NUM_FREE_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    num_free_blocks: int
    def __init__(self, num_free_blocks: _Optional[int] = ...) -> None: ...

class AllocateBlocksRequest(_message.Message):
    __slots__ = ("block_hashes",)
    BLOCK_HASHES_FIELD_NUMBER: _ClassVar[int]
    block_hashes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, block_hashes: _Optional[_Iterable[str]] = ...) -> None: ...

class AllocateBlocksResponse(_message.Message):
    __slots__ = ("success", "block_ids", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    block_ids: _containers.RepeatedScalarFieldContainer[int]
    message: str
    def __init__(self, success: bool = ..., block_ids: _Optional[_Iterable[int]] = ..., message: _Optional[str] = ...) -> None: ...

class FreeBlockRequest(_message.Message):
    __slots__ = ("block_id",)
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    block_id: int
    def __init__(self, block_id: _Optional[int] = ...) -> None: ...

class FreeBlockResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...
