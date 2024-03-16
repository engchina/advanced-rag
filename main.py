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

select_sql = """
    SELECT 
        L2_DISTANCE(:1, :2) euclidean_score0,
        COSINE_DISTANCE(:1, :2) cosine_score0,
        INNER_PRODUCT(:1, :2) dot_score0,
        L2_DISTANCE(:3, :4) euclidean_score1,
        COSINE_DISTANCE(:3, :4) cosine_score1,
        INNER_PRODUCT(:3, :4) dot_score1,
        L2_DISTANCE(:5, :6) euclidean_score2,
        COSINE_DISTANCE(:5, :6) cosine_score2,
        INNER_PRODUCT(:5, :6) dot_score2
    FROM 
        DUAL
"""


def calc_scores(sentence1, sentence2, sentence3):
    # embedding
    cohere_embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    # sentence1_result = cohere_embeddings.embed_query(sentence1)
    # sentence2_result = cohere_embeddings.embed_query(sentence2)
    # sentence3_result = cohere_embeddings.embed_query(sentence3)

    sentences = pd.DataFrame({'text':
        [
            sentence1,
            sentence2,
            sentence3
        ]})

    embeddings = cohere_embeddings.embed_documents(sentences['text'].tolist())
    print(f"{embeddings=}")
    sentence1_result = embeddings[0]
    sentence2_result = embeddings[1]
    sentence3_result = embeddings[2]

    # calculate
    cursor = conn.cursor()
    cursor.setinputsizes(oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR,
                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR,
                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR,
                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR,
                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR,
                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR)
    cursor.execute(select_sql,
                   [sentence1_result, sentence1_result, sentence1_result, sentence1_result, sentence1_result,
                    sentence1_result, sentence1_result, sentence2_result, sentence1_result,
                    sentence2_result, sentence1_result, sentence2_result, sentence1_result, sentence3_result,
                    sentence1_result, sentence3_result, sentence1_result, sentence3_result])

    scores = []
    for row in cursor:
        scores = list(row)
    cursor.close()

    # return (gr.Textbox(value=scores[0]), gr.Textbox(value=scores[1]), gr.Textbox(value=scores[2]),
    #         gr.Textbox(value=scores[3]), gr.Textbox(value=scores[4]), gr.Textbox(value=scores[5]),
    #         gr.Textbox(value=scores[6]), gr.Textbox(value=scores[7]), gr.Textbox(value=scores[8]),
    #         gr.Plot(value=umap_plot(sentences,
    #                                 embeddings)))
    return (gr.Textbox(value=scores[0]), gr.Textbox(value=scores[1]), gr.Textbox(value=scores[2]),
            gr.Textbox(value=scores[3]), gr.Textbox(value=scores[4]), gr.Textbox(value=scores[5]),
            gr.Textbox(value=scores[6]), gr.Textbox(value=scores[7]), gr.Textbox(value=scores[8]),
            )


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
            euclidean_score0 = gr.Textbox(label=f"Euclidean Distance 0")
        with gr.Column(scale=20):
            cosine_score0 = gr.Textbox(label=f"Cosine Similarity 0")
        with gr.Column(scale=20):
            dot_score0 = gr.Textbox(label=f"Dot Product Similarity 0")
    with gr.Row():
        num_scores = 2
        with gr.Column(scale=40):
            sentences = [gr.Textbox(label=f"Sentence {i}") for i in range(1, num_scores + 1)]
        with gr.Column(scale=20):
            euclidean_scores = [gr.Textbox(label=f"Euclidean Distance {i}") for i in range(1, num_scores + 1)]
        with gr.Column(scale=20):
            cosine_scores = [gr.Textbox(label=f"Cosine Similarity {i}") for i in range(1, num_scores + 1)]
        with gr.Column(scale=20):
            dot_scores = [gr.Textbox(label=f"Dot Product Similarity {i}") for i in range(1, num_scores + 1)]
    with gr.Row():
        plot = gr.Plot(label="UMAP", show_label=False, visible=False)
    submit = gr.Button("Submit", variant="primary")
    submit.click(calc_scores,
                 inputs=[sentence0] + sentences,
                 # outputs=[euclidean_score0, cosine_score0,
                 #          dot_score0] + euclidean_scores + cosine_scores + dot_scores + [plot])
                 outputs=[euclidean_score0, cosine_score0,
                          dot_score0] + euclidean_scores + cosine_scores + dot_scores)

app.queue()
if __name__ == "__main__":
    app.launch(show_api=False)
