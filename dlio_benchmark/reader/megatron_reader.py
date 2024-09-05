"""
   Copyright (c) 2024, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import logging

import numpy as np
import struct

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.common.enumerations import DataLoaderSampler
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile

from typing import List, Optional, Tuple, Type, Union
from enum import Enum
import time

dlp = Profile(MODULE_DATA_READER)

_INDEX_HEADER = b"MMIDIDX\x00\x00"

class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[np.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[np.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[np.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[np.number]: The dtype
        """
        return getattr(np, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[np.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[np.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif np.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[np.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[np.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return np.uint16
        else:
            return np.int32

class MegatronReader(FormatReader):
    """
    Reader for Indexed Binary Memory mapped files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch):
        super().__init__(dataset_type, thread_index)
        self.sequence_pointers = {}
        self.sequence_lengths = {}
        self.sequence_count = {}
        self.document_count = {}

        for filename in self._args.id_to_path.values():
            self.init_index_file(filename)

        self.buffer_map = {}

    def index_file_path(self, prefix_path):
        return prefix_path + '.idx'


    def init_index_file(self, filename: str) -> None:
        idx_path = self.index_file_path(filename)
        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count[idx_path] = struct.unpack("<Q", stream.read(8))[0]
            self.document_count[idx_path] = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()
        bin_buffer_mmap = np.memmap(idx_path, mode="r", order="C")
        bin_buffer = memoryview(bin_buffer_mmap)

        logging.info(f"\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths[idx_path] = np.frombuffer(
            bin_buffer, dtype=np.int32, count=self.sequence_count[idx_path], offset=offset
        )
        t_end = time.time()
        logging.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logging.info(f"\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers[idx_path] = np.frombuffer(
            bin_buffer,
            dtype=np.int64,
            count=self.sequence_count[idx_path],
            offset=offset + self.sequence_lengths[idx_path].nbytes,
        )
        t_end = time.time()
        logging.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logging.info(f"\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = np.frombuffer(
            bin_buffer,
            dtype=np.int64,
            count=self.document_count[idx_path],
            offset=offset + self.sequence_lengths[idx_path].nbytes + self.sequence_pointers[idx_path].nbytes,
        )
        t_end = time.time()
        logging.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")


        logging.info(f"> total number of sequences: {self.sequence_count[idx_path]}")
        logging.info(f"> total number of documents: {self.document_indices.shape[0] - 1}")

    def init_index_file_from_disk(self, filename: str) -> None:
            idx_path = self.index_file_path(filename)
            with open(idx_path, "rb") as stream:
                header = stream.read(9)
                assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

                version = struct.unpack("<Q", stream.read(8))[0]
                assert version == 1, f"bad version, cannot read: {idx_path}"

                code = struct.unpack("<B", stream.read(1))[0]
                self.dtype = DType.dtype_from_code(code)
                self.dtype_size = DType.size(self.dtype)

                self.sequence_count[idx_path] = struct.unpack("<Q", stream.read(8))[0]
                self.document_count[idx_path] = struct.unpack("<Q", stream.read(8))[0]

                offset = stream.tell()

            logging.info(f"\tExtract the sequence lengths")
            t_beg = time.time()
            self.sequence_lengths[idx_path] = np.fromfile(
                idx_path, dtype=np.int32, count=self.sequence_count[idx_path], offset=offset
            )
            t_end = time.time()
            logging.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            logging.info(f"\tExtract the sequence pointers")
            t_beg = time.time()
            self.sequence_pointers[idx_path] = np.fromfile(
                idx_path,
                dtype=np.int64,
                count=self.sequence_count[idx_path],
                offset=offset + self.sequence_lengths[idx_path].nbytes,
            )
            t_end = time.time()
            logging.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            logging.info(f"\tExtract the document indices")
            t_beg = time.time()
            self.document_indices = np.fromfile(
                idx_path,
                dtype=np.int64,
                count=self.document_count[idx_path],
                offset=offset + self.sequence_lengths[idx_path].nbytes + self.sequence_pointers[idx_path].nbytes,
            )
            t_end = time.time()
            logging.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")


            logging.info(f"> total number of sequences: {self.sequence_count[idx_path]}")
            logging.info(f"> total number of documents: {self.document_indices.shape[0] - 1}")




    @dlp.log
    def open(self, filename):
        super().open(filename)
        bin_buffer_mmap = np.memmap(filename, mode='r', order='C')
        bin_buffer = memoryview(bin_buffer_mmap)
        self.buffer_map[filename] = np.frombuffer(bin_buffer, dtype=np.uint8)
        return bin_buffer_mmap
    
    # @dlp.log
    # def open(self, filename):
    #     super().open(filename)
    #     return open(filename, "rb")

    @dlp.log
    def close(self, filename):
        super().close(filename)
        self.open_file_map[filename]._mmap.close()

    # @dlp.log
    # def close(self, filename):
    #     super().close(filename)
    #     self.open_file_map[filename].close()

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        buffer = self.buffer_map[filename]
        idx_path = self.index_file_path(filename)
        pointer = self.sequence_pointers[idx_path][sample_index]
        length = self.sequence_lengths[idx_path][sample_index]
        sequence = np.frombuffer(
            buffer,
            dtype=self.dtype,
            count=length,
            offset=pointer
        )
        return sequence
    
    # @dlp.log
    # def get_sample(self, filename, sample_index):
    #     super().get_sample(filename, sample_index)

    #     idx_path = self.index_file_path(filename)
    #     pointer = self.sequence_pointers[idx_path][sample_index]
    #     length = self.sequence_lengths[idx_path][sample_index]
        
    #     sequence = np.empty(length, dtype=self.dtype)
    #     bin_buffer_file = self.open_file_map[filename]
    #     bin_buffer_file.seek(pointer)
    #     bin_buffer_file.readinto(sequence)
    #     return sequence


    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()

    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True