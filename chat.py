import os

from dotenv import find_dotenv, load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

chat = ChatAnthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, model_name="claude-3-opus-20240229")

import gradio as gr


def echo(message, history):
    system = (
        "You are a helpful assistant."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", message)])

    chain = prompt | chat
    response = ""
    for chunk in chain.stream({}):
        print(chunk.content, end="", flush=True)
        response += chunk.content
        yield response


custom_css = """
footer > .svelte-16bt5n8 {
  visibility: hidden
}
"""

app = gr.ChatInterface(fn=echo, title="Claude Opus", css=custom_css)
app.queue()
if __name__ == "__main__":
    app.launch(show_api=False)
