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

TRAIN_PATH = './samples/samples_train.txt'
TEST_PATH = './samples/samples_test.txt'

metadata_cache = CachedMetadataReader(maxsize=100)

myiter = CodeNetIter(language='C++', limit=None)
# train, test 
pids_seen = (defaultdict(int), defaultdict(int))
TRAIN_PBS_PER_PID = 5
TEST_PBS_PER_PID = 2
dataset = [
    [], # train 
    []  # test 
]
for x in myiter:
    x: CodeNetSolution
    x_meta = metadata_cache.read_meta(x.metadata_path, x.submission_id)
    accepted = x_meta is not None and x_meta.status == 'Accepted'
    if not accepted:
        continue
    if pids_seen[x.problem_id][0] >= TRAIN_PBS_PER_PID and pids_seen[x.problem_id][1] >= TEST_PBS_PER_PID:
        continue

    out = try_compile_cpp(src_path=x.src_path)
    if out.returncode != 0:
        continue
    
    dataset_idx = 0 if pids_seen[x.problem_id][0] >= TRAIN_PBS_PER_PID else 1 
    pids_seen[x.problem_id][dataset_idx] += 1

with open(TRAIN_PATH, 'w') as train_file:
    train_file.writelines(dataset[0])

with open(TEST_PATH, 'w') as test_file:
    test_file.writelines(dataset[1])