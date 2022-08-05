MILVUS_HOST = "10.221.197.6"
# MILVUS_HOST = "10.220.127.31"
MILVUS_PORT = 19530
# MILVUS_SERVER_NAME = "my-release-milvus"
MILVUS_COLLECTION_NAME = "sift1m_data_test"

# Does the data need to be normalized before insertion
IF_NORMALIZE = False

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
METRIC_TYPE = 'L2'
num_query_vector, DIM = 1, 128  # DIM ≡ 0 (mod PQ_M)

# Index IVF parameters
NLIST = 2048
NPROBE = 16
PQ_M = 16  # DIM ≡ 0 (mod PQ_M)

# Index NSG parameters
SEARCH_LENGTH = 45
OUT_DEGREE = 50
CANDIDATE_POOL = 300
KNNG = 100


# Index HNSW parameters
HNSW_M = 16
EFCONSTRUCTION = 500
EF = 1

# Index ANNOY parameters
N_TREE = 8
SEARCH_K = 1

NQ_SCOPE = [1, 10, 100, 500, 1000, 2000]
TOPK_SCOPE = [1, 10, 100, 500]

PERCENTILE_NUM = 100

# Location of file, if it is a csv, and if it is stored as UINT8
PERFORMANCE_RESULTS_PATH = 'performance'
QUERY_FILE_PATH = 'milvus_sift1m/query.npy'
BASE_FILE_PATH = 'milvus_sift1m/bvecs_data'

IS_CSV = False
IS_UINT8 = False