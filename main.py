import os

import altair as alt
import gradio as gr
import oracledb
import pandas as pd
import umap
from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import CohereEmbeddings

# read local .env file
_ = load_dotenv(find_dotenv())

conn = oracledb.connect(dsn=os.environ["ORACLE_AI_VECTOR_CONNECTION_STRING"])

# select_sql = """
#     SELECT
#         L2_DISTANCE(:1, :2) euclidean_score0,
#         COSINE_DISTANCE(:3, :4) cosine_score0,
#         INNER_PRODUCT(:5, :6) dot_score0,
#         L2_DISTANCE(:7, :8) euclidean_score1,
#         L2_DISTANCE(:9, :10) euclidean_score2,
#         COSINE_DISTANCE(:11, :12) cosine_score1,
#         COSINE_DISTANCE(:13, :14) cosine_score2,
#         INNER_PRODUCT(:15, :16) dot_score1,
#         INNER_PRODUCT(:17, :18) dot_score2
#     FROM
#         DUAL
# """

# select_sql1 = """
#     SELECT
#         L2_DISTANCE(:1, :2) euclidean_score1,
#         COSINE_DISTANCE(:3, :4) cosine_score1,
#         INNER_PRODUCT(:5, :6) dot_score1
#     FROM
#         DUAL
# """
#
# select_sql2 = """
#     SELECT
#         L2_DISTANCE(:1, :2) euclidean_score2,
#         COSINE_DISTANCE(:3, :4) cosine_score2,
#         INNER_PRODUCT(:5, :6) dot_score2
#     FROM
#         DUAL
# """


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


def calc_scores(sentence0, sentence1, sentence2):
    # embedding
    cohere_embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    sentences_all = pd.DataFrame({'text':
        [
            sentence0,
            sentence1,
            sentence2
        ]})

    embeddings = cohere_embeddings.embed_documents(sentences_all['text'].tolist())
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


def umap_plot(text, emb):
    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    # dense_embeddings = emb.toarray()
    reducer = umap.UMAP(n_neighbors=10)
    umap_embeds = reducer.fit_transform(emb)
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=
        alt.X('x',
              scale=alt.Scale(zero=False)
              ),
        y=
        alt.Y('y',
              scale=alt.Scale(zero=False)
              ),
        tooltip=cols
    ).properties(
        width=700,
        height=400
    )
    return chart


custom_css = """
.gradio-container .gradio-container-4-21-0 {
  font-family: Arial, sans-serif;
}

.app.svelte-182fdeq.svelte-182fdeq {
  max-width: 1920px;
}

footer > .svelte-16bt5n8 {
  visibility: hidden
}
"""

with gr.Blocks(css=custom_css) as app:
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
        plot = gr.Plot(label="UMAP", show_label=False, visible=False)
    submit = gr.Button("Submit", variant="primary")
    submit.click(calc_scores,
                 inputs=[sentence0] + sentences,
                 # outputs=[euclidean_score0, cosine_score0,
                 #          dot_score0] + euclidean_scores + cosine_scores + dot_scores + [plot])
                 outputs=[euclidean_score0, cosine_score0, dot_score0, cosine_similarity0] +
                         euclidean_scores + cosine_scores + dot_scores + cosine_similarities)

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

app.queue()
if __name__ == "__main__":
    app.launch(show_api=False)
