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
# Add or update the vectors for all data values in a table
#

import array
import os
import sys
import time

import oracledb
from FlagEmbedding import BGEM3FlagModel
from dotenv import find_dotenv, load_dotenv
from fastembed import TextEmbedding

# read local .env file
_ = load_dotenv(find_dotenv())

un = os.getenv("PYTHON_USERNAME")
pw = os.getenv("PYTHON_PASSWORD")
cs = os.getenv("PYTHON_CONNECTSTRING")

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

print("Using FastEmbed " + embedding_model)

if embedding_model == "BAAI/bge-m3":
    model = BGEM3FlagModel(embedding_model, use_fp16=True)
else:
    model = TextEmbedding(model_name=embedding_model, max_length=512)

if __name__ == '__main__':

    # Connect to Oracle Database 23.4
    with oracledb.connect(user=un, password=pw, dsn=cs) as connection:
        db_version = tuple(int(s) for s in connection.version.split("."))[:2]
        if db_version < (23, 4):
            sys.exit("This example requires Oracle Database 23.4 or later")
        print("Connected to Oracle Database\n")

        with connection.cursor() as cursor:
            print("Vectorizing the following data:\n")

            # Loop over the rows and vectorize the VARCHAR2 data
            sql = """select id, info
                     from my_data
                     order by 1"""

            binds = []
            tic = time.perf_counter()
            for id_val, info in cursor.execute(sql):
                # Convert to input string format for Sentence Transformers
                data = f"[ 'query: {info}' ]"

                print(info)

                # Create the embedding and extract the vector
                if embedding_model == "BAAI/bge-m3":
                    embedding = model.encode(data,
                                             batch_size=1,
                                             max_length=8192,
                                             # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                             )['dense_vecs']
                    vec = embedding.tolist()
                else:
                    embedding = list(model.embed(data))

                    # Convert to a list
                    vec = list(embedding[0])

                # Convert to array format
                vec2 = array.array("f", vec)

                # Record the array and key
                binds.append([vec2, id_val])

            toc = time.perf_counter()

            # Do an update to add or replace the vector values
            cursor.executemany(
                """update my_data set v = :1
                   where id = :2""",
                binds,
            )
            connection.commit()

            print(f"Vectors took {toc - tic:0.4f} seconds")
            print(f"\nAdded {len(binds)} vectors to the table")
