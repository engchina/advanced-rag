# -----------------------------------------------------------------------------
# Copyright (c) 2023, Oracle and/or its affiliates.
#
# This software is dual-licensed to you under the Universal Permissive License
# (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl and Apache License
# 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose
# either license.
#
# If you elect to accept the software under the Apache License, Version 2.0,
# the following applies:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
#
# Basic Similarity Search using the FastEmbed embedding model
#

import array
import os
import sys
import time

import oracledb
from FlagEmbedding import BGEM3FlagModel, LayerWiseFlagLLMReranker
from dotenv import find_dotenv, load_dotenv
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder

_ = load_dotenv(find_dotenv())

un = os.getenv("PYTHON_USERNAME")
pw = os.getenv("PYTHON_PASSWORD")
cs = os.getenv("PYTHON_CONNECTSTRING")

# topK is how many rows to return
topK = 5

# Re-ranking is about potentially improving the order of the resultset
# Re-ranking is significantly slower than doing similarity search
# Re-ranking is optional
rerank = 1

# English embedding models
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# embedding_model = "nomic-ai/nomic-embed-text-v1"
# embedding_model = "BAAI/bge-small-en-v1.5"
# embedding_model = "BAAI/bge-base-en-v1.5"
# embedding_model = "BAAI/bge-large-en-v1.5"
# embedding_model = "BAAI/bge-large-en-v1.5-quantized"
# embedding_model = "jinaai/jina-embeddings-v2-base-en"
# embedding_model = "jinaai/jina-embeddings-v2-small-en"

# Multi-lingual embedding models
# embedding_model = "intfloat/multilingual-e5-large"
embedding_model = "BAAI/bge-m3"  # not supported by FastEmbed
# embedding_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# English re-rankers
# rerank_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
# rerank_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
# rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# rerank_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# rerank_model = "BAAI/bge-reranker-base"
# rerank_model = "BAAI/bge-reranker-large"

# Multi-lingual re-rankers
# rerank_model = "BAAI/bge-reranker-v2-m3"
rerank_model = "BAAI/bge-reranker-v2-minicpm-layerwise"
# rerank_model = "jeffwan/mmarco-mMiniLMv2-L12-H384-v1"
# rerank_model = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"

print("Using FastEmbed embedding model " + embedding_model)
if rerank:
    print("Using reranker " + rerank_model)

print("TopK = " + str(topK))

sql = """select info
         from my_data
         order by vector_distance(v, :1, COSINE)
         fetch approx first :2 rows only"""

# Define the specific model to use
if embedding_model == "BAAI/bge-m3":
    model = BGEM3FlagModel(embedding_model, use_fp16=True)
else:
    model = TextEmbedding(model_name=embedding_model, max_length=512)

if rerank_model == 'BAAI/bge-reranker-v2-minicpm-layerwise':
    ce = LayerWiseFlagLLMReranker(rerank_model,
                                  use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
else:
    ce = CrossEncoder(rerank_model, max_length=512)

if __name__ == '__main__':
    # Connect to Oracle Database 23.4
    with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
        db_version = tuple(int(s) for s in connection.version.split("."))[:2]
        if db_version < (23, 4):
            sys.exit("This example requires Oracle Database 23.4 or later")
        print("Connected to Oracle Database\n")

        with connection.cursor() as cursor:
            while True:
                # Get the input text to vectorize
                text = input("\nEnter a phrase. Type quit to exit : ")

                if (text == "quit") or (text == "exit"):
                    break

                if text == "":
                    continue

                tic = time.perf_counter()

                # Create the embedding and extract the vector
                # Create the embedding and extract the vector
                if embedding_model == "BAAI/bge-m3":
                    embedding = model.encode(text,
                                             batch_size=1,
                                             max_length=8192,
                                             # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                             )['dense_vecs']

                    toc = time.perf_counter()
                    print(f"Vectorize query took {toc - tic:0.3f} seconds")

                    vec = embedding.tolist()
                else:
                    embedding = list(model.embed(text))

                    toc = time.perf_counter()
                    print(f"Vectorize query took {toc - tic:0.3f} seconds")

                    # Convert to a list
                    vec = list(embedding[0])

                # Convert to array format
                vec2 = array.array("f", vec)

                docs = []
                cross = []

                tic = time.perf_counter()

                # Do the Similarity Search
                for (info,) in cursor.execute(sql, [vec2, topK]):

                    # Remember the SQL data resultset
                    docs.append(info)

                    if rerank == 1:
                        # create the query/data pair needed for cross encoding
                        if rerank_model == 'BAAI/bge-reranker-v2-minicpm-layerwise':
                            cross.append((text, info))
                        else:
                            tup = []
                            tup.append(text)
                            tup.append(info)
                            cross.append(tup)

                toc = time.perf_counter()
                print(f"Similarity Search took {toc - tic:0.4f} seconds")

                if rerank == 0:

                    # Just rely on the vector distance for the resultset order
                    print("\nWithout ReRanking")
                    print("=================")
                    for hit in docs:
                        print(hit)

                else:

                    tic = time.perf_counter()

                    # ReRank for better results
                    if rerank_model == 'BAAI/bge-reranker-v2-minicpm-layerwise':
                        # ce = LayerWiseFlagLLMReranker(rerank_model,
                        #                               use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
                        ce_scores = ce.compute_score(cross, batch_size=1, cutoff_layers=[28])
                    else:
                        # ce = CrossEncoder(rerank_model, max_length=512)
                        ce_scores = ce.predict(cross)

                    toc = time.perf_counter()
                    print(f"Rerank took {toc - tic:0.3f} seconds")

                    # Create the unranked list of ce_scores + data
                    unranked = []
                    for idx in range(topK):
                        tup2 = []
                        tup2.append(ce_scores[idx])
                        tup2.append(docs[idx])
                        unranked.append(tup2)

                    print("\nWithout ReRanked results:")
                    print("=================")
                    for idx in range(topK):
                        x = unranked[idx]
                        print(x[1])

                    # Create the reranked list by sorting the unranked list
                    reranked = sorted(unranked, key=lambda foo: foo[0], reverse=True)

                    print("\nReRanked results:")
                    print("=================")
                    for idx in range(topK):
                        x = reranked[idx]
                        print(x[1])
