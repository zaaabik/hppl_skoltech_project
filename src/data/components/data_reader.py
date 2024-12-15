from abc import ABC
from typing import Dict, Tuple, List

import pyarrow as pa
from lz4 import frame
import os
from tqdm import tqdm
import io
from src.utils.utils import get_hash


class DataReader(ABC):
    """
    Args:
        data_path (str): path data in
    """
    data_paths: Tuple[str]
    indexes: Tuple
    hash_sum: Dict[str, str]

    def __init__(self, data_paths: Tuple[str]) -> None:
        self.data_paths = data_paths

    def setup(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> Dict:
        raise NotImplementedError

    def get_indexes(self):
        raise NotImplementedError


# RAM!!!
class DiskFileDataReader(DataReader):
    file_path: Tuple[str]
    hash_sum: Dict[str, str] = {}

    def setup(self):
        file_paths = []
        index_paths = []
        for path in self.data_paths:
            file_paths.append(os.path.join(path, 'data.txt'))
            index_paths.append(os.path.join(path, 'index.txt'))

        self.file_path = file_paths

        self.file_in_bytes = []
        for file_path in self.file_path:
            with open(file_path, 'rb') as inp:
                self.file_in_bytes.append(io.BytesIO(inp.read()))

        indexes = []
        for index_file in tqdm(index_paths):
            with open(index_file, 'rb') as inp:
                index_bytes = inp.read()

                data_folder_name = os.path.basename(os.path.split(index_file)[0])
                self.hash_sum[data_folder_name] = get_hash(index_bytes)

                indexes.append(DiskFileDataReader.decompress_and_deserialize(index_bytes).copy())
        self.indexes = indexes

    def __getitem__(self, idx) -> Dict:
        if idx not in self:
            raise KeyError
        file = self.get_file_n(idx)
        position, n_bytes = self.indexes[file][idx]

        # with open(self.file_path[file], 'rb') as inp:
        self.file_in_bytes[file].seek(position, 0)
        data = self.file_in_bytes[file].read(n_bytes)

        return dict(DiskFileDataReader.decompress_and_deserialize(data))

    def get_indexes(self):
        all_files_indexes = []
        for file_idxes in self.indexes:
            all_files_indexes.extend(file_idxes)
        return all_files_indexes

    def get_file_n(self, index):
        for ix, i in enumerate(self.indexes):
            if index in i:
                return ix
        return None

    def __contains__(self, idx):
        for i in self.indexes:
            if idx in i:
                return True
        return False

    @staticmethod
    def decompress_and_deserialize(data):
        return pa.deserialize(frame.decompress(data))
