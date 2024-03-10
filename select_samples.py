'''
Just a simple script to select a subset of submissions to run our experiments. Currenly select:
- Accepted submissions 
- One per problem id 
- Ones that compile (not all accepted submissions compile with our args as the user may have chosen a different C++ compiler)
- At most 1000 different pids 
'''
from collections import defaultdict 

from codenet_data import CodeNetIter, CodeNetSolution
from codenet_utils import CachedMetadataReader
from compiler_utils import try_compile_cpp

metadata_cache = CachedMetadataReader(maxsize=100)

myiter = CodeNetIter(language='C++', limit=None)
pids_seen = defaultdict(int)
MAX_PIDS = 5000
MAX_PBS_PER_PID = 5
for x in myiter:
    x: CodeNetSolution
    x_meta = metadata_cache.read_meta(x.metadata_path, x.submission_id)
    accepted = x_meta is not None and x_meta.status == 'Accepted'
    if accepted and pids_seen[x.problem_id] < MAX_PBS_PER_PID:
        out = try_compile_cpp(src_path=x.src_path)
        if out.returncode != 0:
            continue

        pids_seen[x.problem_id] += 1
        print(x.src_path)

    if len(pids_seen) > MAX_PIDS:
        break
