'''
Just a simple script to select a subset of submissions to run our experiments. Currenly select:
- Accepted submissions 
- One per problem id 
- Ones that compile (not all accepted submissions compile with our args as the user may have chosen a different C++ compiler)
- At most 1000 different pids 
'''

from codenet_data import CodeNetIter, CodeNetSolution
from codenet_utils import CachedMetadataReader
from compiler_utils import try_compile_cpp

metadata_cache = CachedMetadataReader(maxsize=100)

myiter = CodeNetIter(language='C++', limit=None)
pids_seen = set()
MAX_PIDS = 1000
for x in myiter:
    x: CodeNetSolution
    x_meta = metadata_cache.read_meta(x.metadata_path, x.submission_id)
    if x_meta is not None and x_meta.status == 'Accepted' and x.problem_id not in pids_seen:
        out = try_compile_cpp(path=x.src_path)
        if out.returncode != 0:
            continue

        pids_seen.add(x.problem_id)
        print(x.src_path)

    if len(pids_seen) > MAX_PIDS:
        break