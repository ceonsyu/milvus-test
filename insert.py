import os
import time

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION_NAME, DIM, BASE_FILE_PATH
from helper import count, load_npy_data

fmt = "\n=== {:30} ===\n"
num_partition = 5
num_entities_per_partition, dim = 200000, DIM
num_entities_per_load = 5000
search_latency_fmt = "search latency = {:.4f}s"

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format(f"start connecting to Milvus"))
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

#utility.drop_collection(MILVUS_COLLECTION_NAME)  # drop existing collection
has = utility.has_collection(MILVUS_COLLECTION_NAME)
print(f"Does collection {MILVUS_COLLECTION_NAME} exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 2 fields.
print(fmt.format(f"creating collection {MILVUS_COLLECTION_NAME}"))
fields = [
    FieldSchema(name="sku_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="feature", dtype=DataType.FLOAT_VECTOR, dim=dim)
]
schema = CollectionSchema(fields, "search test")
print(fmt.format(f"Create collection {MILVUS_COLLECTION_NAME}"))
collection = Collection(MILVUS_COLLECTION_NAME, schema, consistency_level="Strong")
has = utility.has_collection(MILVUS_COLLECTION_NAME)
print(f"Does collection {MILVUS_COLLECTION_NAME} exist in Milvus: {has}")

################################################################################
# 3. insert data
# We are going to insert 1 rows of data into `test_milvus`
print(fmt.format("Start inserting entities"))

# rng = np.random.default_rng(seed=19530)
# num = num_entities_per_load
# for i in range(num_partition):
#     partition = Partition(test_milvus, f"partiton{i}", "")
#     while num <= (i+1)*num_entities_per_partition:
#         if num%5000==0:
#             print(f"current num of entities:{num}")
#         entities = [
#             # provide the pk field because `auto_id` is set to False
#             [i for i in range(num-num_entities_per_load, num)],
#             -1+2*rng.random((num_entities_per_load, dim)),    # field feature, supports numpy.ndarray and list
#         ]
#         partition.insert(entities)
#         num += num_entities_per_load
#     print(f"num of entities in partition{i}:{partition.num_entities}")

#############
# SIFT_DATA #
filenames = os.listdir(BASE_FILE_PATH)
filenames.sort()
total_insert_time = 0
collection_rows = count(collection)
for filename in filenames:
    vectors = load_npy_data(os.path.join(BASE_FILE_PATH, filename))
    vectors_ids = list(id for id in range(collection_rows, collection_rows + len(vectors)))
    # vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
    time_add_start = time.time()
    collection.insert([vectors_ids, vectors])
    total_insert_time = total_insert_time + time.time() - time_add_start
    print(filename, "insert rows", len(vectors_ids), " insert milvus time: ", time.time() - time_add_start)
    collection_rows = collection_rows + len(vectors_ids)
count(collection)
print("total insert time: ", total_insert_time)


collection.load()
print(f"Number of entities in {MILVUS_COLLECTION_NAME}: {collection.num_entities}")  # check the num_entites

print(fmt.format("Start querying with `sku_id<10`"))

result = collection.query(expr="sku_id<10", output_fields=["sku_id", "feature"])

print(f"query result:\n-{result[0]}")
