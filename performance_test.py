import datetime
import os

from config import PERFORMANCE_RESULTS_PATH, MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME
from helper import connect, search, create_index

test_milvus = connect(MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME)

if not os.path.exists(PERFORMANCE_RESULTS_PATH):
    os.mkdir(PERFORMANCE_RESULTS_PATH)
now = datetime.datetime.now()
result_filename =now.strftime("%Y-%m-%d %H:%M:%S") + '_performance.csv'
performance_file = os.path.join(PERFORMANCE_RESULTS_PATH, result_filename)

with open(performance_file, 'w+', encoding='utf-8') as f:
    # IVF_FLAT
    index_type = 'IVF_FLAT'
    create_index(test_milvus, index_type)
    search(test_milvus, index_type, f)
    # 'IVF_PQ'
    # index_type = 'IVF_PQ'
    # create_index(test_milvus, index_type)
    # search(test_milvus, index_type, f)
