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

from enum import Enum
from typing import List, Optional, Tuple, Type, Union
from types import TracebackType
from dlio_benchmark.common.enumerations import Compression
from dlio_benchmark.data_generator.data_generator import DataGenerator

import logging
import numpy as np
import torch

from dlio_benchmark.utils.utility import progress, utcnow, DLIOMPI
from dlio_benchmark.utils.utility import Profile
from shutil import copyfile
from dlio_benchmark.common.constants import MODULE_DATA_GENERATOR
import struct
from mpi4py import MPI

dlp = Profile(MODULE_DATA_GENERATOR)

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


class _IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[np.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: Type[np.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(
        self,
        sequence_lengths: List[int],
        document_indices: List[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            document_indices (List[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)   # 用于定位到每个句子的开头，偏移量以字节为单位

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = np.array(document_indices, dtype=np.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))


    def _sequence_pointers(self, sequence_lengths: List[int]) -> List[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += length * itemsize
        return list_ptr
    
class IndexedDatasetBuilder(object):
    """Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[numpy.number], optional): The dtype of the index file. Defaults to numpy.int32.

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
        self, bin_path: str, dtype: Type[np.number] = np.int32, multimodal: bool = False
    ) -> None:
        self.data_file = open(bin_path, "wb")
        self.dtype = dtype
        self.multimodal = multimodal

        self.sequence_lengths = []  # 保存每个句子的长度
        self.document_indices = [0] # 保存每个doc即json行文本的起始位置
        self.sequence_modes = [] if self.multimodal else None

    def add_item(self, tensor: torch.Tensor, mode: int = 0) -> None:
        """Add a single item to the dataset

        Args:
            tensor (torch.Tensor): The item to add to the data file

            mode (int, optional): The mode for the item. Defaults to 0.
        """
        np_array = np.array(tensor.numpy(), dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)
        if self.multimodal:
            self.sequence_modes.append(mode)

    def add_document(
        self, tensor: torch.Tensor, lengths: List[int], modes: Optional[List[int]] = None
    ) -> None:
        """Add an entire document to the dataset

        Args:
            tensor (torch.Tensor): The document to add

            lengths (List[int]): The lengths of each item in the document

            modes (Optional[List[int]], optional): The modes for each item in the document. Defaults to None.
        """
        np_array = np.array(tensor, dtype=self.dtype)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.extend(lengths)
        self.document_indices.append(len(self.sequence_lengths))    # doc之间的偏移量为句子的个数
        if self.multimodal:
            self.sequence_modes.extend(modes if modes is not None else [0] * lengths)

    def end_document(self) -> None:
        """Finalize the document, for use with IndexedDatasetBuilder.add_item"""
        self.document_indices.append(len(self.sequence_lengths))

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.sequence_modes, self.document_indices)

"""
Generator for creating data in NPZ format.
"""
class MegatronGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.dtype = np.uint8

    def index_file_path(self, prefix_path):
        return prefix_path + '.idx'

    def index_file_path_off(self, prefix_path):
        return prefix_path + '.off.idx'

    def index_file_path_size(self, prefix_path):
        return prefix_path + '.sz.idx'

    @dlp.log
    def generate(self):
        """
        Generator for creating data in NPZ format of 3d dataset.
        """
        super().generate()
        np.random.seed(10)
        MB=1048576
        samples_processed = 0
        total_samples = self.total_files_to_generate * self.num_samples
        dim = self.get_dimension(self.total_files_to_generate)
        logging.info(dim)
        if self.total_files_to_generate <= self.comm_size:
            # Use collective I/O
            # we need even number os samples for collective I/O
            samples_per_rank = (self.num_samples + (self.num_samples % self.comm_size)) // self.comm_size
            for file_index in dlp.iter(range(int(self.total_files_to_generate))):
                amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
                comm = MPI.COMM_WORLD
                dim1 = dim[2*file_index]
                dim2 = dim[2*file_index + 1]
                sample_size = dim1 * dim2
                out_path_spec = self.storage.get_uri(self._file_list[file_index])
                # out_path_spec_off_idx = self.index_file_path_off(out_path_spec)
                # out_path_spec_sz_idx = self.index_file_path_size(out_path_spec)
                out_path_spec_idx = self.index_file_path(out_path_spec)
                fh = MPI.File.Open(comm, out_path_spec, amode)
                samples_per_loop = int(MB / sample_size)

                for sample_index in range(self.my_rank*samples_per_rank, samples_per_rank*(self.my_rank+1), samples_per_loop):
                    #logging.info(f"{utcnow()} rank {self.my_rank} writing {sample_index} * {samples_per_loop} for {samples_per_rank} samples")
                    records = records = np.random.randint(255, size=sample_size*samples_per_loop, dtype=np.uint8)
                    offset = sample_index * sample_size
                    fh.Write_at_all(offset, records)
                    samples_processed += samples_per_loop
                    progress(samples_processed * self.comm_size, total_samples, "Generating Indexed Binary Data Samples")
                fh.Close()
                logging.info(f"{utcnow()} rank {self.my_rank} writing metadata")
                # off_file = open(out_path_spec_off_idx, "wb")
                # sz_file = open(out_path_spec_sz_idx, "wb")
                
                if int(file_index / self.comm_size) == self.my_rank:
                    # # Write offsets
                    # myfmt = 'Q' * self.num_samples
                    # data_to_write = self.num_samples * sample_size
                    # samples_to_write = self.num_samples
                    # offsets = range(0, data_to_write, sample_size)
                    # offsets = offsets[:samples_to_write]
                    # binary_offsets = struct.pack(myfmt, *offsets)
                    # off_file.write(binary_offsets)

                    # # Write sizes
                    # myfmt = 'Q' * samples_to_write
                    # sample_sizes = [sample_size] * samples_to_write
                    # binary_sizes = struct.pack(myfmt, *sample_sizes)
                    # sz_file.write(binary_sizes)

                    # write index file
                    document_count = self.num_files_train
                    samples_per_doc = self.num_samples
                    sequence_count = self.num_samples

                    self.document_indices = [samples_per_doc] * document_count
                    self.sequence_lengths = [sample_size] * sequence_count

                    with _IndexWriter(out_path_spec_idx, self.dtype) as writer:
                        writer.write(self.sequence_lengths, self.document_indices)

        else:
            for i in dlp.iter(range(self.my_rank, int(self.total_files_to_generate), self.comm_size)):
                dim1 = dim[2*i]
                dim2 = dim[2*i + 1]
                sample_size = dim1 * dim2
                total_size = sample_size * self.num_samples
                write_size = total_size
                memory_size = self._args.generation_buffer_size
                if total_size > memory_size:
                    write_size = memory_size - (memory_size % sample_size)
                out_path_spec = self.storage.get_uri(self._file_list[i])
                out_path_spec_off_idx = self.index_file_path_off(out_path_spec)
                out_path_spec_sz_idx = self.index_file_path_size(out_path_spec)
                out_path_spec_idx = self.index_file_path(out_path_spec)
                progress(i + 1, self.total_files_to_generate, "Generating Indexed Binary Data")
                prev_out_spec = out_path_spec
                written_bytes = 0
                data_file = open(out_path_spec, "wb")
                # off_file = open(out_path_spec_off_idx, "wb")
                # sz_file = open(out_path_spec_sz_idx, "wb")
                records = np.random.randint(255, size=write_size, dtype=np.uint8)
                while written_bytes < total_size:
                    data_to_write = write_size if written_bytes + write_size <= total_size else total_size - written_bytes
                    samples_to_write = data_to_write // sample_size

                    # Write data
                    myfmt = 'B' * data_to_write
                    binary_data = struct.pack(myfmt, *records[:data_to_write])
                    data_file.write(binary_data)
                    struct._clearcache()

                    written_bytes = written_bytes + data_to_write
                
                # write index file
                document_count = self.num_files_train
                samples_per_doc = self.num_samples
                sequence_count = self.num_samples

                self.document_indices = [samples_per_doc] * document_count
                self.sequence_lengths = [sample_size] * sequence_count

                with _IndexWriter(out_path_spec_idx, self.dtype) as writer:
                    writer.write(self.sequence_lengths, self.document_indices)


                data_file.close()

            np.random.seed()
        DLIOMPI.get_instance().comm().Barrier()
