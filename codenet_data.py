from pathlib import Path
import math
from functools import cached_property, lru_cache
from typing import Optional, List
from dataclasses import dataclass, field

from queue import SimpleQueue

from bs4 import BeautifulSoup
from langdetect import detect

CODENET_ROOT = Path('./Project_CodeNet/')

@dataclass 
class CodeNetSolution:
    problem_id: str 
    submission_id: str 
    language: str 
    src_path: Path 
    dataset_root: Path = field(default=CODENET_ROOT)
        
    def assert_exists(self):
        assert self.metadata_path.exists()
        assert self.pdescr_path.exists()
        assert self.src_path.exists()
    
    @classmethod 
    def from_path(cls, path: Path):
        submission_id = path.stem
        lang_folder = path.parent
        language = lang_folder.name 
        pb_folder = lang_folder.parent
        problem_id = pb_folder.name 
        return cls(
            src_path=path,
            submission_id=submission_id,
            language=language,
            problem_id=problem_id
        )
    
    def with_root(self, new_root: Path):
        src_rel = self.src_path.relative_to(self.dataset_root)
        new_src = new_root.joinpath(src_rel)
        return CodeNetSolution(
            problem_id=self.problem_id,
            submission_id=self.submission_id,
            language=self.language,
            src_path=new_src,
            dataset_root=new_root
        )
        
    @property
    def metadata_path(self) -> Path:
        return self.dataset_root.joinpath('metadata').joinpath(f'{self.problem_id}.csv')
    
    @property 
    def pdescr_path(self) -> Path:
        return self.dataset_root.joinpath('problem_descriptions').joinpath(f'{self.problem_id}.html')        
        
class CodeNetIter:
    @cached_property 
    def data_root(self) -> Path:
        return CODENET_ROOT
    
    def __init__(self, language: str, limit: Optional[int] = None, pde_lang: str = "en"):
        self.limit = limit or math.inf 
        self.language = language
        self.pde_lang = pde_lang

        self._cursor = 0
        self._pb_iter = None  
        self._src_iter = None 
        self._done = False 

    def check_language(self, result: CodeNetSolution) -> bool:
        """Checks if solution's problem description is in target language

        Args:
            result (CodeNetSolution): solution object to be checked

        Returns:
            bool: True is language matches self.pde_lang, False otherwise
        """
        with(result.pdescr_path.open('r') as file):
            try:
                soup = BeautifulSoup(file, 'html.parser')
                text_content = ' '.join(element.get_text(strip=True) for element in soup.find_all('p')) 
                detected_language = detect(text_content) 
            except IOError as e:
                print(f"Error opening or reading file: {e}")
                return False 
            except BeautifulSoup.ParserRejectedMarkup as e:
                print(f"Error parsing HTML: {e}")
                return False
            except langdetect.lang_detect_exception.LangDetectException as e:
                print(f"Language detection error: {e}")
                return False

        return self.pde_lang == detected_language
    
    @cached_property
    def src_root(self) -> Path:
        return self.data_root.joinpath('data')
    
    def _advance(self):
        if self._cursor == self.limit:
            self._done = True 
            return 
        
        if self._pb_iter is None:
            self._pb_iter = self.src_root.iterdir()
            
        if self._src_iter is None:
            lang_folder = None 
            while lang_folder is None or not lang_folder.exists():
                try: 
                    pb_folder = next(self._pb_iter)    
                except StopIteration:
                    self._done = True 
                    return 

                lang_folder = pb_folder.joinpath(self.language)

            self._src_iter = lang_folder.iterdir()        
    
    def _consume_value(self):
        while True:
            try:
                while True:
                    try:
                        path = next(self._src_iter)
                        # Make sure metadata and pdescr exist 
                        result = CodeNetSolution.from_path(path)
                        result.assert_exists()

                        #Skip problems that don't match target language
                        #if(not self.check_language(result)):
                            #continue
                        return result
                    except AssertionError:
                        continue                         
            except StopIteration:
                self._src_iter = None 
                self._advance()
                if self._done:
                    raise StopIteration
    
    def __iter__(self):            
        self._advance()
        return self 
    
    def __next__(self):
        if self._done:
            raise StopIteration
        
        value = self._consume_value()
        self._cursor += 1
        self._advance()
        return value