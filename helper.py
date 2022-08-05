import os
import sys
import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    Collection, FieldSchema, DataType, CollectionSchema,
)
from config import fmt, METRIC_TYPE, HNSW_M, EFCONSTRUCTION, N_TREE, NLIST, PQ_M, \
    DIM, NPROBE, SEARCH_K, EF, SEARCH_LENGTH, NQ_SCOPE, TOPK_SCOPE, BASE_FILE_PATH, MILVUS_HOST, MILVUS_COLLECTION_NAME, \
    MILVUS_PORT, QUERY_FILE_PATH


def get_search_params(index_type):
    if index_type == 'FLAT':
        search_params = {"metric_type": METRIC_TYPE}
    elif index_type == 'RNSG':
        search_params = {"metric_type": METRIC_TYPE, "params": {'search_length': SEARCH_LENGTH}}
    elif index_type == 'HNSW':
        search_params = {"metric_type": METRIC_TYPE, "params": {'ef': EF}}
    elif index_type == 'ANNOY':
        search_params = {"metric_type": METRIC_TYPE, "params": {"search_k": SEARCH_K}}
    else:
        search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": NPROBE}}
    return search_params


def get_index_params(index_type):
    if index_type == 'FLAT':
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": {}}
    elif index_type == 'HNSW':
        params = {"M": HNSW_M, "efConstruction": EFCONSTRUCTION}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    elif index_type == 'ANNOY':
        params = {"n_trees": N_TREE}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    elif index_type == 'IVF_PQ':
        params = {"nlist": NLIST, "m": PQ_M}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    else:
        params = {"nlist": NLIST}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    return index_param


def connect(host, port, name):
    print(fmt.format(f"start connecting to Milvus"))
    connections.connect("default", host=host, port=port)

    has = utility.has_collection(name)
    print(f"Does collection {name} exist in Milvus: {has}")
    return Collection(name)


def get_nq_vec_random(nq):
    rng = np.random.default_rng(seed=19530)
    return -1+2*rng.random((nq, DIM))

def get_nq_vec_from_file(query):
    data = np.load(QUERY_FILE_PATH)
    if len(data) > query:
        return data[0:query].tolist()
    else:
        print(f'There is only {len(data)} vectors')
        return data.tolist()

def create_index(collection, index_type):
    print(fmt.format(f"Start Creating index {index_type}"))
    index_params = get_index_params(index_type)

    time_start = time.time()
    collection.create_index("feature", index_params)
    time_cost = time.time() - time_start
    print(f"time cost of Creating index {index_type}:{round(time_cost, 4)}s")


def search(collection, index_type, f):
    f.write("index_type,nq,topk,total_time/s,avg_time/s,VPS" + '\n')
    search_params = get_search_params(index_type)
    for nq in NQ_SCOPE:
        vectors_to_search = get_nq_vec_from_file(nq)
        for topk in TOPK_SCOPE:
            print(fmt.format(f"begin to search, nq = {len(vectors_to_search)}, topk={topk}"))
            # partiton = collection.partition("partiton0")
            time_start = time.time()
            results = collection.search(vectors_to_search, "feature", search_params, limit=topk)
            # partiton.search(vectors_to_search, "feature", search_params, limit=topk)
            time_cost = time.time() - time_start
            line = str(index_type) + ',' + str(nq) + ',' + str(topk) + ',' + str(round(time_cost, 4)) + ',' + str(
                round(time_cost / nq, 4)) + ',' + str(round(nq / time_cost, 4)) + '\n'
            f.write(line)
            print(f"search done!,time cost:{round(time_cost, 4)}")


def load_npy_data(filename):
    data = np.load(filename)
    # if IS_UINT8:
    #     data = (data + 0.5) / 255
    # if IF_NORMALIZE:
    #     data = normalize(data)
    # data = data.tolist()
    return data


def npy_to_milvus():
    filenames = os.listdir(BASE_FILE_PATH)
    filenames.sort()
    total_insert_time = 0
    collection = connect(MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME)
    collection_rows = count(collection)
    for filename in filenames:
        vectors = load_npy_data(os.path.join(BASE_FILE_PATH, filename))
        vectors_ids = list(id for id in range(collection_rows, collection_rows + len(vectors)))
        # vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
        time_add_start = time.time()
        ids = collection.insert(MILVUS_COLLECTION_NAME, vectors, vectors_ids)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(filename, "insert rows", len(ids), " insert milvus time: ", time.time() - time_add_start)
        collection_rows = collection_rows + len(ids)
    count(collection)
    print("total insert time: ", total_insert_time)


def count(collection):
    # Get the number of milvus collection
    try:
        num = collection.num_entities
        print(f"Successfully get the num:{num} of the collection:{MILVUS_COLLECTION_NAME}")
        return num
    except Exception as e:
        print(f"Failed to count vectors in Milvus: {e}")
        sys.exit(1)
