import os
import re

import cohere
import gradio as gr
import oracledb
import pandas as pd
from FlagEmbedding import LayerWiseFlagLLMReranker, FlagReranker, FlagLLMReranker
from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mylangchain.embeddings.bge_m3 import BGEM3Embeddings
from mylangchain.embeddings.e5_large import MultilingualE5LargeEmbeddings
from mylangchain.embeddings.e5_large_instruct import MultilingualE5LargeInstructEmbeddings
from mylangchain.vectorstores.oracleaivector import OracleAIVector

# from langchain_community.embeddings import CohereEmbeddings

# read local .env file
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
MY_COMPARTMENT_OCID = os.environ["MY_COMPARTMENT_OCID"]

ORACLE_AI_VECTOR_CONNECTION_STRING = os.environ["ORACLE_AI_VECTOR_CONNECTION_STRING"]
conn = oracledb.connect(dsn=ORACLE_AI_VECTOR_CONNECTION_STRING)

# command_r_chat = ChatOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, model_name="gpt-4",
#                             temperature=0)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)
command_r_chat = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-r", max_tokens=4096, temperature=0)
claude_opus_chat = ChatAnthropic(anthropic_api_key=ANTHROPIC_API_KEY, model_name="claude-3-opus-20240229",
                                 temperature=0)
claude_sonnet_chat = ChatAnthropic(anthropic_api_key=ANTHROPIC_API_KEY, model_name="claude-3-sonnet-20240229",
                                   temperature=0)
google_gemini_pro_chat = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro")

oci_cohere_embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=MY_COMPARTMENT_OCID,
)
bge_m3_embeddings = BGEM3Embeddings()
e5_large_instruct_embeddings = MultilingualE5LargeInstructEmbeddings()
e5_large_embeddings = MultilingualE5LargeEmbeddings()

file_contents = []
file_contents_splits = []

select_sql_euclidean_score0 = """
    SELECT 
        L2_DISTANCE(:1, :2) euclidean_score0
    FROM 
        DUAL
"""

select_sql_cosine_score0 = """
    SELECT 
        COSINE_DISTANCE(:1, :2) cosine_score0
    FROM 
        DUAL
"""

select_sql_dot_score0 = """
    SELECT 
        INNER_PRODUCT(:1, :2) dot_score0
    FROM 
        DUAL
"""

select_sql_euclidean_score1 = """
    SELECT 
        L2_DISTANCE(:1, :2) euclidean_score1
    FROM 
        DUAL
"""

select_sql_cosine_score1 = """
    SELECT 
        COSINE_DISTANCE(:1, :2) cosine_score1
    FROM 
        DUAL
"""

select_sql_dot_score1 = """
    SELECT 
        INNER_PRODUCT(:1, :2) dot_score1
    FROM 
        DUAL
"""

select_sql_euclidean_score2 = """
    SELECT 
        L2_DISTANCE(:1, :2) euclidean_score2
    FROM 
        DUAL
"""

select_sql_cosine_score2 = """
    SELECT 
        COSINE_DISTANCE(:1, :2) cosine_score2
    FROM 
        DUAL
"""

select_sql_dot_score2 = """
    SELECT 
        INNER_PRODUCT(:1, :2) dot_score2
    FROM 
        DUAL
"""


def load_document(uploaded_file_input):
    """
    Specify a DocumentLoader to load in your unstructured data as Documents.
    A Document is a dict with text (page_content) and metadata.
    """
    loader = PyMuPDFLoader(uploaded_file_input.name)

    global file_contents
    file_contents = loader.load()
    for doc in file_contents:
        file_name = os.path.basename(doc.metadata['source'])
        doc.metadata['source'] = file_name
        doc.metadata['file_path'] = file_name
    print(f"{file_contents=}")
    all_page_content_text_output = ""
    page_count = len(file_contents)
    for i in range(page_count):
        page_content_text_output = re.sub(r'\n\n+', '|' * 10, file_contents[i].page_content)
        page_content_text_output = re.sub(r'\s+', ' ', page_content_text_output)
        page_content_text_output = re.sub(r'\|{10}', '\n', page_content_text_output)
        file_contents[i].page_content = page_content_text_output
        all_page_content_text_output += '\n\n' + page_content_text_output

    return gr.Textbox(value=str(page_count)), gr.Textbox(value=all_page_content_text_output)


def split_document(chunk_size_text_input, chunk_overlap_text_input):
    """
    Split the Document into chunks for embedding and vector storage.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size_text_input),
                                                   chunk_overlap=int(chunk_overlap_text_input),
                                                   separators=["\n\n", "\n", "。", "、", "？", "！", "；", ".", ",", "?",
                                                               "!", ";"])

    global file_contents, file_contents_splits
    file_contents_splits = text_splitter.split_documents(file_contents)
    print(f"{file_contents_splits=}")
    chunk_count_text_output = len(file_contents_splits)
    first_trunk_content_text_output = file_contents_splits[0].page_content
    last_trunk_content_text_output = file_contents_splits[-1].page_content

    return gr.Textbox(value=str(chunk_count_text_output)), gr.Textbox(
        value=first_trunk_content_text_output), gr.Textbox(value=last_trunk_content_text_output)


def embed_document(docs_embedding_model_checkbox_group_input):
    """
    To be able to look up our document splits, we first need to store them where we can later look them up.
    The most common way to do this is to embed the contents of each document split.
    We store the embedding and splits in a vectorstore.
    """
    print(f"{docs_embedding_model_checkbox_group_input=}")
    global file_contents_splits
    first_trunk_vector_text_output = bge_m3_embeddings.embed_query(file_contents_splits[0].page_content)
    last_trunk_vector_text_output = bge_m3_embeddings.embed_query(file_contents_splits[-1].page_content)

    # OCI Cohere support inputs array size within [1, 96].
    # For oci cohere
    if "cohere/embed-multilingual-v3.0" in docs_embedding_model_checkbox_group_input:
        pre_delete_collection = True
        oci_cohere_file_contents_splits = file_contents_splits.copy()
        while oci_cohere_file_contents_splits:
            # Process documents in chunks of 96
            current_chunk = oci_cohere_file_contents_splits[:96]
            OracleAIVector.from_documents(
                embedding=oci_cohere_embeddings,
                documents=current_chunk,
                collection_name="docs_oci_cohere",
                connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                pre_delete_collection=pre_delete_collection,
            )
            # Only the first chunk needs to pre-delete the collection.
            pre_delete_collection = False
            # Prepare the remaining documents for the next iteration.
            oci_cohere_file_contents_splits = oci_cohere_file_contents_splits[96:]

    # For bge-m3
    if "BAAI/bge-m3" in docs_embedding_model_checkbox_group_input:
        pre_delete_collection = True
        bge_m3_file_contents_splits = file_contents_splits.copy()
        while bge_m3_file_contents_splits:
            # Process documents in chunks of 96
            current_chunk = bge_m3_file_contents_splits[:96]
            OracleAIVector.from_documents(
                embedding=bge_m3_embeddings,
                documents=current_chunk,
                collection_name="docs_bge_m3",
                connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                pre_delete_collection=pre_delete_collection,
            )
            # Only the first chunk needs to pre-delete the collection.
            pre_delete_collection = False
            # Prepare the remaining documents for the next iteration.
            bge_m3_file_contents_splits = bge_m3_file_contents_splits[96:]

    # For multilingual-e5-large-instruct
    if "intfloat/multilingual-e5-large-instruct" in docs_embedding_model_checkbox_group_input:
        pre_delete_collection = True
        e5_large_instruct_file_contents_splits = file_contents_splits.copy()
        while e5_large_instruct_file_contents_splits:
            # Process documents in chunks of 96
            current_chunk = e5_large_instruct_file_contents_splits[:96]
            OracleAIVector.from_documents(
                embedding=e5_large_instruct_embeddings,
                documents=current_chunk,
                collection_name="docs_e5_large_instruct",
                connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                pre_delete_collection=pre_delete_collection,
            )
            # Only the first chunk needs to pre-delete the collection.
            pre_delete_collection = False
            # Prepare the remaining documents for the next iteration.
            e5_large_instruct_file_contents_splits = e5_large_instruct_file_contents_splits[96:]

    # For multilingual-e5-large
    if "intfloat/multilingual-e5-large" in docs_embedding_model_checkbox_group_input:
        pre_delete_collection = True
        e5_large_file_contents_splits = file_contents_splits.copy()
        while e5_large_file_contents_splits:
            # Process documents in chunks of 96
            current_chunk = e5_large_file_contents_splits[:96]
            OracleAIVector.from_documents(
                embedding=e5_large_embeddings,
                documents=current_chunk,
                collection_name="docs_e5_large",
                connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                pre_delete_collection=pre_delete_collection,
            )
            # Only the first chunk needs to pre-delete the collection.
            pre_delete_collection = False
            # Prepare the remaining documents for the next iteration.
            e5_large_file_contents_splits = e5_large_file_contents_splits[96:]

    return gr.Textbox(value=str(first_trunk_vector_text_output)), gr.Textbox(value=str(last_trunk_vector_text_output))


def chat_document_stream(question_embedding_model_checkbox_group_input, reranker_model_radio_input,
                         llm_answer_checkbox_group_input, question_text_input):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """
    print(f"{question_embedding_model_checkbox_group_input=}")
    # Use OracleAIVector
    vectorstores = []
    unranked_docs = []
    # For oci cohere
    if "cohere/embed-multilingual-v3.0" in question_embedding_model_checkbox_group_input:
        oci_cohere_vectorstore = OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                                                collection_name="docs_oci_cohere",
                                                embedding_function=oci_cohere_embeddings,
                                                )
        vectorstores.append(oci_cohere_vectorstore)

    # For bge-m3
    if "BAAI/bge-m3" in question_embedding_model_checkbox_group_input:
        bge_m3_vectorstore = OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                                            collection_name="docs_bge_m3",
                                            embedding_function=bge_m3_embeddings,
                                            )
        vectorstores.append(bge_m3_vectorstore)

    # For e5-large-instruct
    if "intfloat/multilingual-e5-large-instruct" in question_embedding_model_checkbox_group_input:
        e5_large_vectorstore = OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                                              collection_name="docs_e5_large_instruct",
                                              embedding_function=e5_large_instruct_embeddings,
                                              )
        vectorstores.append(e5_large_vectorstore)

    # For e5-large
    if "intfloat/multilingual-e5-large" in question_embedding_model_checkbox_group_input:
        e5_large_vectorstore = OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                                              collection_name="docs_e5_large",
                                              embedding_function=e5_large_embeddings,
                                              )
        vectorstores.append(e5_large_vectorstore)

    top_k = 10
    for vectorstore in vectorstores:
        docs = vectorstore.similarity_search_with_score(question_text_input, k=top_k)
        unranked_docs.extend(docs)

    if reranker_model_radio_input == 'cohere/rerank-multilingual-v2.0':
        unranked = []
        for doc, _ in unranked_docs:
            unranked.append(doc.page_content)
        ranked_results = cohere_client.rerank(query=question_text_input,
                                              documents=unranked,
                                              top_n=len(unranked_docs),
                                              model='rerank-multilingual-v2.0')

        ranked_scores = [0.0] * len(unranked_docs)
        for result in ranked_results.results:
            ranked_scores[result.index] = result.relevance_score

        docs_data = [{'ページ・コンテンツ': doc.page_content,
                      'コサイン距離': score,
                      'ソース': doc.metadata["source"],
                      'score': ce_score} for (doc, score), ce_score in zip(unranked_docs, ranked_scores)]
        docs_dataframe = pd.DataFrame(docs_data)

        docs_dataframe = docs_dataframe.sort_values(by='score', ascending=False).head(top_k).drop(
            columns=['score'])
    elif reranker_model_radio_input == 'BAAI/bge-reranker-v2-minicpm-layerwise':
        cross = [(question_text_input, doc.page_content) for doc, _ in unranked_docs]
        ce = LayerWiseFlagLLMReranker(reranker_model_radio_input, use_fp16=True)
        ce_scores = ce.compute_score(cross, batch_size=1, cutoff_layers=[28])

        docs_data = [{'ページ・コンテンツ': doc.page_content,
                      'コサイン距離': score,
                      'ソース': doc.metadata["source"],
                      'score': ce_score} for (doc, score), ce_score in zip(unranked_docs, ce_scores)]
        docs_dataframe = pd.DataFrame(docs_data)

        docs_dataframe = docs_dataframe.sort_values(by='score', ascending=False).head(top_k).drop(
            columns=['score'])
    elif reranker_model_radio_input == 'BAAI/bge-reranker-v2-gemma':
        cross = [(question_text_input, doc.page_content) for doc, _ in unranked_docs]
        ce = FlagLLMReranker('BAAI/bge-reranker-v2-gemma',
                             use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
        ce_scores = ce.compute_score(cross, batch_size=1)

        docs_data = [{'ページ・コンテンツ': doc.page_content,
                      'コサイン距離': score,
                      'ソース': doc.metadata["source"],
                      'score': ce_score} for (doc, score), ce_score in zip(unranked_docs, ce_scores)]
        docs_dataframe = pd.DataFrame(docs_data)

        docs_dataframe = docs_dataframe.sort_values(by='score', ascending=False).head(top_k).drop(
            columns=['score'])
    elif reranker_model_radio_input == 'BAAI/bge-reranker-v2-m3':
        cross = [(question_text_input, doc.page_content) for doc, _ in unranked_docs]
        ce = FlagReranker('BAAI/bge-reranker-v2-m3',
                          use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
        ce_scores = ce.compute_score(cross, batch_size=1)

        docs_data = [{'ページ・コンテンツ': doc.page_content,
                      'コサイン距離': score,
                      'ソース': doc.metadata["source"],
                      'score': ce_score} for (doc, score), ce_score in zip(unranked_docs, ce_scores)]
        docs_dataframe = pd.DataFrame(docs_data)

        docs_dataframe = docs_dataframe.sort_values(by='score', ascending=False).head(top_k).drop(
            columns=['score'])
    else:
        docs_data = [{'ページ・コンテンツ': doc.page_content,
                      'コサイン距離': score,
                      'ソース': doc.metadata["source"]} for doc, score in unranked_docs]
        docs_dataframe = pd.DataFrame(docs_data)

    command_r_result = ""
    opus_result = ""
    sonnet_result = ""
    gemini_result = ""
    if len(llm_answer_checkbox_group_input) > 0:
        template = """
        Use the following pieces of Context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use the EXACT TEXT from the Context WITHOUT ANY MODIFICATIONS, REORGANIZATION or EMBELLISHMENT.
        Don't try to answer anything that isn't in Context.
        Context:
        ```
        {context}
        ```
        Question: 
        ```
        {question}
        ```
        Helpful Answer:"""
        rag_prompt_custom = PromptTemplate.from_template(template)
        # Method-1
        # retriever = vectorstore.as_retriever()
        #
        # rag_chain = (
        #         {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
        # )
        #
        # result = rag_chain.invoke(question2_text)
        # # print(result)
        #
        # return gr.Dataframe(value=docs_dataframe), gr.Textbox(result.content)

        # Method-2
        message = rag_prompt_custom.format_prompt(context=docs_dataframe['ページ・コンテンツ'].tolist(),
                                                  question=question_text_input)
        print(f"{message.to_messages()=}")
        if "cohere/command-r" in llm_answer_checkbox_group_input:
            command_r_result = command_r_chat.invoke(message.to_messages()).content
        if "claude/opus" in llm_answer_checkbox_group_input:
            opus_result = claude_opus_chat.invoke(message.to_messages()).content
        if "claude/sonnet" in llm_answer_checkbox_group_input:
            sonnet_result = claude_sonnet_chat.invoke(message.to_messages()).content
        if "google/gemini-pro" in llm_answer_checkbox_group_input:
            gemini_result = google_gemini_pro_chat.invoke(message.to_messages()).content

    return gr.Dataframe(value=docs_dataframe, wrap=True, column_widths=["65%", "15%", "20%"]), gr.Textbox(
        command_r_result), gr.Textbox(opus_result), gr.Textbox(sonnet_result), gr.Textbox(
        gemini_result)


def calc_distance_scores(calculate_embedding_model_radio_input, sentence0_input, sentence1_input, sentence2_input):
    # embedding
    # cohere_embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    sentences_all = pd.DataFrame({'text':
        [
            sentence0_input,
            sentence1_input,
            sentence2_input
        ]})

    if calculate_embedding_model_radio_input == 'cohere/embed-multilingual-v3.0':
        embeddings = oci_cohere_embeddings.embed_documents(sentences_all['text'].tolist())
    elif calculate_embedding_model_radio_input == 'BAAI/bge-m3':
        embeddings = bge_m3_embeddings.embed_documents(sentences_all['text'].tolist())
    elif calculate_embedding_model_radio_input == 'intfloat/multilingual-e5-large':
        embeddings = e5_large_embeddings.embed_documents(sentences_all['text'].tolist())
    else:
        embeddings = e5_large_embeddings.embed_documents(sentences_all['text'].tolist())

    sentence0_embedding = embeddings[0]
    sentence1_embedding = embeddings[1]
    sentence2_embedding = embeddings[2]

    scores = []
    # calculate
    cursor = conn.cursor()
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_euclidean_score0, [sentence0_embedding, sentence0_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_cosine_score0, [sentence0_embedding, sentence0_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_dot_score0, [sentence0_embedding, sentence0_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)

    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_euclidean_score1, [sentence0_embedding, sentence1_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_cosine_score1, [sentence0_embedding, sentence1_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_dot_score1, [sentence0_embedding, sentence1_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)

    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_euclidean_score2, [sentence0_embedding, sentence2_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_cosine_score2, [sentence0_embedding, sentence2_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql_dot_score2, [sentence0_embedding, sentence2_embedding])
    for row in cursor:
        # print(f"{row=}")
        scores += list(row)

    cursor.close()

    # return (gr.Textbox(value=scores[0]), gr.Textbox(value=scores[1]), gr.Textbox(value=scores[2]),
    #         gr.Textbox(value=scores[3]), gr.Textbox(value=scores[4]), gr.Textbox(value=scores[5]),
    #         gr.Textbox(value=scores[6]), gr.Textbox(value=scores[7]), gr.Textbox(value=scores[8]),
    #         gr.Plot(value=umap_plot(sentences,
    #                                 embeddings)))
    cosine_similarity0 = ""
    if scores[1] <= 0.14:
        cosine_similarity0 = "似ている(<=0.14)"
    elif scores[1] <= 0.5:
        cosine_similarity0 = "やや似ている(<=0.5)"
    elif scores[1] <= 1:
        cosine_similarity0 = "ほぼ関係ない(<=1)"
    elif scores[1] <= 2:
        cosine_similarity0 = "関係ない(<=2)"

    cosine_similarity1 = ""
    if scores[4] <= 0.14:
        cosine_similarity1 = "似ている(<=0.14)"
    elif scores[4] <= 0.5:
        cosine_similarity1 = "やや似ている(<=0.5)"
    elif scores[4] <= 1:
        cosine_similarity1 = "ほぼ関係ない(<=1)"
    elif scores[4] <= 2:
        cosine_similarity1 = "関係ない(<=2)"

    cosine_similarity2 = ""
    if scores[7] <= 0.14:
        cosine_similarity2 = "似ている(<=0.14)"
    elif scores[7] <= 0.5:
        cosine_similarity2 = "やや似ている(<=0.5)"
    elif scores[7] <= 1:
        cosine_similarity2 = "ほぼ関係ない(<=1)"
    elif scores[7] <= 2:
        cosine_similarity2 = "関係ない(<=2)"

    return (gr.Textbox(value=scores[0]), gr.Textbox(value=scores[1]), gr.Textbox(value=scores[2]),
            gr.Textbox(value=cosine_similarity0),
            gr.Textbox(value=scores[3]), gr.Textbox(value=scores[6]), gr.Textbox(value=scores[4]),
            gr.Textbox(value=scores[7]), gr.Textbox(value=scores[5]), gr.Textbox(value=scores[8]),
            gr.Textbox(value=cosine_similarity1), gr.Textbox(value=cosine_similarity2))


# def umap_plot(text, emb):
#     cols = list(text.columns)
#     # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
#     # dense_embeddings = emb.toarray()
#     reducer = umap.UMAP(n_neighbors=10)
#     umap_embeds = reducer.fit_transform(emb)
#     df_explore = text.copy()
#     df_explore['x'] = umap_embeds[:, 0]
#     df_explore['y'] = umap_embeds[:, 1]
#
#     # Plot
#     chart = alt.Chart(df_explore).mark_circle(size=60).encode(
#         x=
#         alt.X('x',
#               scale=alt.Scale(zero=False)
#               ),
#         y=
#         alt.Y('y',
#               scale=alt.Scale(zero=False)
#               ),
#         tooltip=cols
#     ).properties(
#         width=700,
#         height=400
#     )
#     return chart


custom_css = """
.gradio-container .gradio-container-4-22-0 {
  font-family: Arial, sans-serif;
}

.app.svelte-182fdeq.svelte-182fdeq {
  max-width: 1800px;
}

footer > .svelte-16bt5n8 {
  visibility: hidden
}
"""

with gr.Blocks(css=custom_css) as app:
    gr.Markdown(value="# Advanced RAG 評価システム")

    with gr.Tabs() as tabs:
        with gr.TabItem(label="Step-1.ドキュメントの読み込み"):
            with gr.Row():
                with gr.Column():
                    page_count_text = gr.Textbox(label="ページ数", lines=1)
            with gr.Row():
                with gr.Column():
                    page_content_text = gr.Textbox(label="ページ内容", lines=15, max_lines=15, autoscroll=False,
                                                   show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    uploaded_file = gr.File(label="ファイル", file_types=[".pdf", ".html", ".docx", ".pptx"],
                                            type="filepath")
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[os.path.join(os.path.dirname(__file__), "files/glossary-ref-en.pdf"),
                                          os.path.join(os.path.dirname(__file__), "files/glossary-ref-ja.pdf"),
                                          os.path.join(os.path.dirname(__file__), "files/glossary-ref-zh.pdf")],
                                label="ファイル事例",
                                inputs=uploaded_file)
            with gr.Row():
                with gr.Column():
                    load_button = gr.Button(value="読み込み", variant="primary")

        with gr.TabItem(label="Step-2.ドキュメントの分割"):
            with gr.Row():
                with gr.Column():
                    chunk_count_text = gr.Textbox(label="Chunk 数", lines=1)
            with gr.Row():
                with gr.Column():
                    first_trunk_content_text = gr.Textbox(label="最初の Chunk 内容", lines=10, max_lines=10,
                                                          autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    last_trunk_content_text = gr.Textbox(label="最後の Chunk 内容", lines=10, max_lines=10,
                                                         autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    chunk_size_text = gr.Textbox(label="チャンク・サイズ(Chunk Size)", lines=1, value="500")
                with gr.Column():
                    chunk_overlap_text = gr.Textbox(label="チャンク・オーバーラップ(Chunk Overlap)", lines=1,
                                                    value="100")
            with gr.Row():
                with gr.Column():
                    gr.Examples(
                        examples=[[50, 0], [200, 0], [500, 0], [500, 100], [1000, 200], [2000, 200], [4000, 400],
                                  [8000, 800]],
                        inputs=[chunk_size_text, chunk_overlap_text])
            with gr.Row():
                with gr.Column():
                    split_button = gr.Button(value="分割", variant="primary")

        with gr.TabItem(label="Step-3.ベクトル化&データベースへの保存"):
            with gr.Row():
                with gr.Column():
                    first_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                         autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    last_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                        autoscroll=False, show_copy_button=True)

            with gr.Row():
                # Model intfloat/multilingual-e5-large-instruct is not supported in TextEmbedding.
                # Please check the supported models using `TextEmbedding.list_supported_models()`
                # docs_embedding_model_checkbox_group = gr.CheckboxGroup(
                #     ["cohere/embed-multilingual-v3.0", "BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct",
                #      "intfloat/multilingual-e5-large"],
                #     label="Embedding モデル")
                docs_embedding_model_checkbox_group = gr.CheckboxGroup(
                    ["cohere/embed-multilingual-v3.0", "BAAI/bge-m3",
                     "intfloat/multilingual-e5-large"],
                    label="Embedding モデル*", value="cohere/embed-multilingual-v3.0")
            with gr.Row():
                with gr.Column():
                    embed_and_save_button = gr.Button(value="ベクトル化して保存", variant="primary")

        with gr.TabItem(label="Step-4.ドキュメントとチャット"):
            with gr.Row():
                with gr.Column():
                    result_dataframe = gr.Dataframe(
                        headers=["ページ・コンテンツ", "コサイン距離", "ソース"],
                        datatype=["str", "str", "str"],
                        row_count=5,
                        col_count=(3, "fixed"),
                        wrap=True,
                        column_widths=["65%", "15%", "20%"]
                    )
            with gr.Row():
                with gr.Column():
                    answer_by_cohere_command_r_text = gr.Textbox(label="Cohere Command-r 回答", lines=5, max_lines=10,
                                                                 autoscroll=False, interactive=False,
                                                                 show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    answer_by_claude_opus_text = gr.Textbox(label="Claude Opus 回答", lines=5, max_lines=10,
                                                            autoscroll=False, interactive=False, show_copy_button=True)

            with gr.Row():
                with gr.Column():
                    answer_by_claude_sonnet_text = gr.Textbox(label="Claude Sonnet 回答", lines=5, max_lines=10,
                                                              autoscroll=False, interactive=False,
                                                              show_copy_button=True)

            with gr.Row():
                with gr.Column():
                    answer_by_google_gemini_text = gr.Textbox(label="Google Gemini Pro 回答", lines=5, max_lines=10,
                                                              autoscroll=False, interactive=False,
                                                              show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    # Model intfloat/multilingual-e5-large-instruct is not supported in TextEmbedding.
                    # Please check the supported models using `TextEmbedding.list_supported_models()`
                    # question_embedding_model_checkbox_group = gr.CheckboxGroup(
                    #     ["cohere/embed-multilingual-v3.0", "BAAI/bge-m3", "intfloat/multilingual-e5-large-instruct",
                    #      "intfloat/multilingual-e5-large"],
                    #     label="Embedding モデル")
                    question_embedding_model_checkbox_group = gr.CheckboxGroup(
                        ["cohere/embed-multilingual-v3.0", "BAAI/bge-m3",
                         "intfloat/multilingual-e5-large"],
                        label="Embedding モデル*", value="cohere/embed-multilingual-v3.0")
            with gr.Row():
                with gr.Column():
                    reranker_model_radio = gr.Radio(
                        ["None", "cohere/rerank-multilingual-v2.0",
                         "BAAI/bge-reranker-v2-minicpm-layerwise",
                         "BAAI/bge-reranker-v2-gemma",
                         "BAAI/bge-reranker-v2-m3"],
                        label="Reranker モデル*", value="cohere/rerank-multilingual-v2.0")
            with gr.Row():
                with gr.Column():
                    llm_answer_checkbox_group = gr.CheckboxGroup(
                        ["cohere/command-r", "claude/opus", "claude/sonnet", "google/gemini-pro"],
                        label="LLM 回答", value="claude/opus")
            with gr.Row():
                with gr.Column():
                    question_text = gr.Textbox(label="質問*", lines=1)
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["Kendraとは？",
                                          "着信トラフィックを分散させるには？",
                                          "ディストリビューションとは？",
                                          "動画をエンコードするに",
                                          "Security Groupとは？",
                                          "深層学習推論のコスト削減率は？",
                                          "リージョンに存在するアベイラビリティゾーンの数は？",
                                          "アペリケーション",
                                          "What is Kendra?",
                                          "How to distribute incoming traffic?",
                                          "What is distribution?",
                                          "How do I encode a video?",
                                          "What is security groups?",
                                          "What is the cost reduction rate for deep learning inference?",
                                          "How many availability zones exist in the region?",
                                          "applcation",
                                          ],
                                inputs=question_text)
            with gr.Row():
                with gr.Column():
                    chat_document_button = gr.Button(value="送信", variant="primary")

        with gr.TabItem(label="Step-5.距離の計算"):
            with gr.Row():
                with gr.Column(scale=40):
                    sentence0 = gr.Textbox(label="文字列-0")
                with gr.Column(scale=20):
                    cosine_score0 = gr.Textbox(label=f"コサイン距離-0")
                with gr.Column(scale=20):
                    cosine_similarity0 = gr.Textbox(label=f"コサイン類似度-0")
                with gr.Column(scale=20, visible=False):
                    euclidean_score0 = gr.Textbox(label=f"Euclidean Distance 0")
                with gr.Column(scale=20, visible=False):
                    dot_score0 = gr.Textbox(label=f"Dot Product Similarity 0")
            with gr.Row():
                num_scores = 2
                with gr.Column(scale=40):
                    sentences = [gr.Textbox(label=f"文字列-{i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20):
                    cosine_scores = [gr.Textbox(label=f"コサイン距離-{i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20):
                    cosine_similarities = [gr.Textbox(label=f"コサイン類似度-{i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20, visible=False):
                    euclidean_scores = [gr.Textbox(label=f"Euclidean Distance {i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20, visible=False):
                    dot_scores = [gr.Textbox(label=f"Dot Product Similarity {i}") for i in range(1, num_scores + 1)]
            with gr.Row():
                calculate_embedding_model_radio = gr.Radio(
                    ["cohere/embed-multilingual-v3.0", "BAAI/bge-m3",
                     "intfloat/multilingual-e5-large"],
                    label="Embedding モデル*", value="cohere/embed-multilingual-v3.0")
            with gr.Row():
                # plot = gr.Plot(label="UMAP", show_label=False, visible=False)
                calculate_distance = gr.Button("計算", variant="primary")

            with gr.Row():
                gr.Markdown(value="## Reference: Vector Distance Metrics")
            with gr.Row():
                with gr.Column(scale=50):
                    gr.Markdown(value="### Cosine Similarity")
                    gr.Image(value="./images/cosine.png", height=750, width=750)
                    gr.Markdown(
                        value="コサイン類似度とは、2 つのベクトルの間の角度のコサイン値を計算したものです。角度が小さいほど、2 つのベクトルの類似度が高くなります。")
                with gr.Column(scale=50):
                    gr.Markdown(value="### Euclidean Distances")
                    gr.Image(value="./images/euclidean.png", height=750, width=750)
                    gr.Markdown(
                        value="ユークリッド距離は、比較しているベクトルの各対応する座標間の距離、つまり2点間の直線距離を表します。")
                with gr.Column(scale=40, visible=False):
                    gr.Markdown(value="### Dot Product Similarity")
                    gr.Image(value="./images/dot.png", height=750, width=750)
                    gr.Markdown(
                        value="大きなドット積の値は、ベクトル同士が類似していることを示唆し、小さな値はあまり類似していないことを示唆します。")
                    gr.Markdown(
                        value="ユークリッド距離を使用するよりも、ドット積による類似度を使用することは、特に高次元ベクトルに対して有用です。")
                    gr.Markdown(
                        value="ベクトルを正規化し、ドット積による類似度を使用することは、コサイン類似度を使用することと同等であることに注意してください。")

        load_button.click(load_document,
                          inputs=[uploaded_file],
                          outputs=[page_count_text, page_content_text],
                          )

        split_button.click(split_document,
                           inputs=[chunk_size_text, chunk_overlap_text],
                           outputs=[chunk_count_text, first_trunk_content_text, last_trunk_content_text]
                           )

        embed_and_save_button.click(embed_document,
                                    inputs=[docs_embedding_model_checkbox_group],
                                    outputs=[first_trunk_vector_text, last_trunk_vector_text],
                                    )

        chat_document_button.click(chat_document_stream,
                                   inputs=[question_embedding_model_checkbox_group, reranker_model_radio,
                                           llm_answer_checkbox_group, question_text],
                                   outputs=[result_dataframe, answer_by_cohere_command_r_text,
                                            answer_by_claude_opus_text, answer_by_claude_sonnet_text,
                                            answer_by_google_gemini_text])

    calculate_distance.click(calc_distance_scores,
                             inputs=[calculate_embedding_model_radio] + [sentence0] + sentences,
                             # outputs=[euclidean_score0, cosine_score0,
                             #          dot_score0] + euclidean_scores + cosine_scores + dot_scores + [plot])
                             outputs=[euclidean_score0, cosine_score0, dot_score0,
                                      cosine_similarity0] + euclidean_scores + cosine_scores + dot_scores + cosine_similarities)

app.queue()
if __name__ == "__main__":
    app.launch(show_api=False, server_port=7861)
