
from pathlib import Path
import math
from functools import cached_property, lru_cache
from typing import Optional, List
from dataclasses import dataclass, field
from queue import SimpleQueue

@dataclass 
class SolutionMeta:
    submission_id: str
    status: str 
    code_size: int 

class EfficientMetadataFile:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._fd = open(file_path, 'r')
        self._file_iter = iter(self._fd.readlines())
        # Skip header 
        next(self._file_iter)
        self._cache = dict()
    
    def _parse_line(self, line: str) -> SolutionMeta:
        elements = line.strip('\n').split(',')
        code_size = int(elements[-2])
        status = elements[7]
        submission_id = elements[0]
        return SolutionMeta(
            submission_id=submission_id,
            status=status,
            code_size=code_size
        )
    
    def _try_find(self, submission_id: str):
        assert submission_id not in self._cache
        while True:
            try:
                next_meta = self._parse_line(next(self._file_iter))
                self._cache[next_meta.submission_id] = next_meta
                if next_meta.submission_id == submission_id:
                    break 
            except StopIteration:
                return
        
    def get_meta(self, submission_id: str) -> Optional[SolutionMeta]:
        if submission_id not in self._cache:
            self._try_find(submission_id)
        
        return self._cache.get(submission_id, None)
    
    def close(self):
        self._fd.close()

class CachedMetadataReader:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        
        # Have to implement the lru_cache ourselves since we need to call .close() on eviction 
        self._queue = SimpleQueue()
        # Path -> (File, time)
        self._cache = dict()
        self._time = 0
        
    def _pop_cache(self):
        while True:
            top_path, top_time = self._queue.get()
            cached_file, cached_time = self._cache[top_path]
            if cached_time == top_time:
                cached_file.close()
                del self._cache[top_path]
                return 
        
    def _add_to_cache(self, file: EfficientMetadataFile):
        if len(self._cache) == self.maxsize:
            self._pop_cache()
        
        self._cache[file.file_path] = (file, self._time)
        self._queue.put((file.file_path, self._time))
        self._time += 1
        
    def _read_file(self, file_path: Path) -> EfficientMetadataFile:
        if file_path in self._cache:
            file = self._cache[file_path][0]
        else:
            file = EfficientMetadataFile(file_path)
        
        self._add_to_cache(file)
        return file 
    
    def read_meta(self, file_path: Path, submission_id: str):
        file = self._read_file(file_path)
        return file.get_meta(submission_id)
    
@lru_cache(maxsize=1000)
def read_src_file(file_path: Path):
    with open(file_path, 'r') as f:
        return f.readlines()