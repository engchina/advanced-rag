import os
import re

import gradio as gr
import oracledb
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
# from langchain_community.embeddings import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mylangchain.embeddings.bge_m3 import BGEM3Embeddings
from mylangchain.vectorstores.oracleaivector import OracleAIVector

# read local .env file
_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
ORACLE_AI_VECTOR_CONNECTION_STRING = os.environ["ORACLE_AI_VECTOR_CONNECTION_STRING"]
conn = oracledb.connect(dsn=ORACLE_AI_VECTOR_CONNECTION_STRING)

command_r_chat = ChatOpenAI(api_key=OPENAI_API_KEY, base_url="", model_name="gpt-4",
                            temperature=0)
bge_m3_embeddings = BGEM3Embeddings()

file_content = []
file_content_splits = []

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

    global file_content
    file_content = loader.load()
    # print(f"data: {data}")
    all_page_content_text_output = ""
    page_count = len(file_content)
    for i in range(page_count):
        page_content_text_output = re.sub(r'\n\n+', '|' * 10, file_content[i].page_content)
        page_content_text_output = re.sub(r'\s+', ' ', page_content_text_output)
        page_content_text_output = re.sub(r'\|{10}', '\n', page_content_text_output)
        file_content[i].page_content = page_content_text_output
        all_page_content_text_output += '\n\n' + page_content_text_output

    return gr.Textbox(value=str(page_count)), gr.Textbox(value=all_page_content_text_output)


def split_document(chunk_size_text_input, chunk_overlap_text_input):
    """
    Split the Document into chunks for embedding and vector storage.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size_text_input),
                                                   chunk_overlap=int(chunk_overlap_text_input))

    global file_content, file_content_splits
    file_content_splits = text_splitter.split_documents(file_content)
    # print(f"all_splits: {all_splits}")
    chunk_count_text_output = len(file_content_splits)
    first_trunk_content_text_output = file_content_splits[0].page_content
    last_trunk_content_text_output = file_content_splits[-1].page_content

    return gr.Textbox(value=str(chunk_count_text_output)), gr.Textbox(
        value=first_trunk_content_text_output), gr.Textbox(value=last_trunk_content_text_output)


def embed_document():
    """
    To be able to look up our document splits, we first need to store them where we can later look them up.
    The most common way to do this is to embed the contents of each document split.
    We store the embedding and splits in a vectorstore.
    """
    first_trunk_vector_text_output = bge_m3_embeddings.embed_query(file_content_splits[0].page_content)
    last_trunk_vector_text_output = bge_m3_embeddings.embed_query(file_content_splits[-1].page_content)

    # Use OracleAIVector
    OracleAIVector.from_documents(
        embedding=bge_m3_embeddings,
        documents=file_content_splits,
        collection_name="docs_of_oracle_ai_vector",
        connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
        pre_delete_collection=True,  # Overriding a vectorstore
    )
    # print(f"vectorstore: {vectorstore}")

    return gr.Textbox(value=str(first_trunk_vector_text_output)), gr.Textbox(value=str(last_trunk_vector_text_output))


def chat_document_stream(question_text_input):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """
    # Use OracleAIVector
    vectorstore = OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                                 collection_name="docs_of_oracle_ai_vector",
                                 embedding_function=bge_m3_embeddings,
                                 )
    docs_dataframe = []
    docs = vectorstore.similarity_search_with_score(question_text_input)
    # print(f"len(docs): {len(docs)}")
    for doc, score in docs:
        # print(f"doc: {doc}")
        # print("Score: ", score)
        docs_dataframe.append([doc.page_content, doc.metadata["source"]])

    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use ten sentences maximum and keep the answer as concise as possible.
    Don't try to answer anything that isn't in context.  
    {context}
    Question: {question}
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
    message = rag_prompt_custom.format_prompt(context=docs, question=question_text_input)
    print(f"message.to_messages(): {message.to_messages()}")
    result = command_r_chat(message.to_messages())
    return gr.Dataframe(value=docs_dataframe, wrap=True, column_widths=["70%", "30%"]), gr.Textbox(result.content)


def calc_distance_scores(sentence0_input, sentence1_input, sentence2_input):
    # embedding
    # cohere_embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    sentences_all = pd.DataFrame({'text':
        [
            sentence0_input,
            sentence1_input,
            sentence2_input
        ]})

    # embeddings = cohere_embeddings.embed_documents(sentences_all['text'].tolist())
    embeddings = bge_m3_embeddings.embed_documents(sentences_all['text'].tolist())
    sentence0_embedding = embeddings[0]
    sentence1_embedding = embeddings[1]
    sentence2_embedding = embeddings[2]
    # print(f"{sentence0_embedding=}")
    # print(f"{sentence1_embedding=}")
    # print(f"{sentence2_embedding=}")

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
    gr.Markdown(value="# Advanced RAG")

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
                    gr.Examples(examples=[[50, 0], [200, 0], [500, 0], [500, 100], [1000, 200]],
                                inputs=[chunk_size_text, chunk_overlap_text])
            with gr.Row():
                with gr.Column():
                    split_button = gr.Button(value="分割", variant="primary")

        with gr.TabItem(label="Step-3.ベクトル・データベースへ保存"):
            with gr.Row():
                with gr.Column():
                    first_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                         autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    last_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                        autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    embed_and_save_button = gr.Button(value="ベクトル化して保存", variant="primary")

        with gr.TabItem(label="Step-4.ドキュメントとチャット"):
            with gr.Row():
                with gr.Column():
                    result_dataframe = gr.Dataframe(
                        headers=["ページ・コンテンツ", "ソース"],
                        datatype=["str", "str"],
                        row_count=5,
                        col_count=(2, "fixed"),
                        wrap=True,
                        column_widths=["70%", "30%"]
                    )
            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(label="回答", lines=15, max_lines=15,
                                             autoscroll=False, interactive=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    question_text = gr.Textbox(label="質問", lines=1)
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["Kendraとは？",
                                          "着信トラフィックを分散させるには？",
                                          "ディストリビューションとは？",
                                          "動画をエンコードするに",
                                          "Security Groupとは？",
                                          "深層学習推論のコスト削減率は？",
                                          "リージョンに存在するアベイラビリティゾーンの数は？",
                                          "アペリケーション"
                                          ],
                                inputs=question_text)
            with gr.Row():
                with gr.Column():
                    chat_document_button = gr.Button(value="送信", variant="primary")

        with gr.TabItem(label="Step-5.距離の計算"):
            with gr.Row():
                with gr.Column(scale=40):
                    sentence0 = gr.Textbox(label="Sentence 0")
                with gr.Column(scale=20):
                    cosine_score0 = gr.Textbox(label=f"Cosine Distance 0")
                with gr.Column(scale=20):
                    cosine_similarity0 = gr.Textbox(label=f"Cosine Similarity 0")
                with gr.Column(scale=20):
                    euclidean_score0 = gr.Textbox(label=f"Euclidean Distance 0")
                with gr.Column(scale=20, visible=False):
                    dot_score0 = gr.Textbox(label=f"Dot Product Similarity 0")
            with gr.Row():
                num_scores = 2
                with gr.Column(scale=40):
                    sentences = [gr.Textbox(label=f"Sentence {i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20):
                    cosine_scores = [gr.Textbox(label=f"Cosine Distance {i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20):
                    cosine_similarities = [gr.Textbox(label=f"Cosine Similarity {i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20):
                    euclidean_scores = [gr.Textbox(label=f"Euclidean Distance {i}") for i in range(1, num_scores + 1)]
                with gr.Column(scale=20, visible=False):
                    dot_scores = [gr.Textbox(label=f"Dot Product Similarity {i}") for i in range(1, num_scores + 1)]
            with gr.Row():
                # plot = gr.Plot(label="UMAP", show_label=False, visible=False)
                calculate_distance = gr.Button("計算", variant="primary")

            with gr.Row():
                gr.Markdown(value="## Reference: Vector Distance Metrics")
            with gr.Row():
                with gr.Column(scale=50):
                    gr.Markdown(value="### Cosine Similarity")
                    gr.Markdown(
                        value="コサイン類似度とは、2 つのベクトルの間の角度のコサイン値を計算したものです。角度が小さいほど、2 つのベクトルの類似度が高くなります。")
                    gr.Image(value="./images/cosine.png")
                with gr.Column(scale=50):
                    gr.Markdown(value="### Euclidean Distances")
                    gr.Markdown(
                        value="ユークリッド距離は、比較しているベクトルの各対応する座標間の距離、つまり2点間の直線距離を表します。")
                    gr.Image(value="./images/euclidean.png")
                with gr.Column(scale=40, visible=False):
                    gr.Markdown(value="### Dot Product Similarity")
                    gr.Markdown(
                        value="大きなドット積の値は、ベクトル同士が類似していることを示唆し、小さな値はあまり類似していないことを示唆します。")
                    gr.Markdown(
                        value="ユークリッド距離を使用するよりも、ドット積による類似度を使用することは、特に高次元ベクトルに対して有用です。")
                    gr.Markdown(
                        value="ベクトルを正規化し、ドット積による類似度を使用することは、コサイン類似度を使用することと同等であることに注意してください。")
                    gr.Image(value="./images/dot.png")

        load_button.click(load_document,
                          inputs=[uploaded_file],
                          outputs=[page_count_text, page_content_text],
                          )

        split_button.click(split_document,
                           inputs=[chunk_size_text, chunk_overlap_text],
                           outputs=[chunk_count_text, first_trunk_content_text, last_trunk_content_text]
                           )

        embed_and_save_button.click(embed_document,
                                    inputs=[],
                                    outputs=[first_trunk_vector_text, last_trunk_vector_text],
                                    )

        chat_document_button.click(chat_document_stream,
                                   inputs=[question_text],
                                   outputs=[result_dataframe, answer_text])

        calculate_distance.click(calc_distance_scores,
                                 inputs=[sentence0] + sentences,
                                 # outputs=[euclidean_score0, cosine_score0,
                                 #          dot_score0] + euclidean_scores + cosine_scores + dot_scores + [plot])
                                 outputs=[euclidean_score0, cosine_score0, dot_score0,
                                          cosine_similarity0] + euclidean_scores + cosine_scores + dot_scores + cosine_similarities)

app.queue()
if __name__ == "__main__":
    app.launch(show_api=False)
