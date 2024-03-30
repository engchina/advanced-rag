import array
from typing import Dict, List, Optional

from fastembed import TextEmbedding
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings


class MultilingualE5LargeEmbeddings(BaseModel, Embeddings):
    """intfloat/multilingual-e5-large embedding models.

    To use, you should have the ``intfloat/multilingual-e5-large`` python package installed, and the
    environment variable ``intfloat/multilingual-e5-large_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import intfloat/multilingual-e5-largeEmbeddings
            intfloat/multilingual-e5-large = intfloat/multilingual-e5-largeEmbeddings(
                model="embed-english-light-v2.0", intfloat/multilingual-e5-large_api_key="my-api-key"
            )
    """
    truncate: Optional[str] = None
    """Truncate embeddings that are too long from start or end ("NONE"|"START"|"END")"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from fastembed import TextEmbedding

        except ImportError:
            raise ValueError(
                "Could not import BGEM3FlagModel python package. "
                "Please install it with `pip install fastembed`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to intfloat/multilingual-e5-large embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embed_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", max_length=512)
        embeddings = list(embed_model.embed(texts))
        return [list(map(float, array.array("f", e))) for e in embeddings]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to intfloat/multilingual-e5-large's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embed_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", max_length=512)
        embeddings = list(embed_model.embed(texts))
        return [list(map(float, array.array("f", e))) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to intfloat/multilingual-e5-large's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embed_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", max_length=512)
        embeddings = list(embed_model.embed([text]))
        return list(map(float, array.array("f", embeddings[0])))

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to intfloat/multilingual-e5-large's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embed_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", max_length=512)
        embeddings = list(embed_model.embed([text]))
        return list(map(float, array.array("f", embeddings[0])))
