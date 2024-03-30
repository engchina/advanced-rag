from typing import Dict, List, Optional

from FlagEmbedding import BGEM3FlagModel
from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings


class BGEM3Embeddings(BaseModel, Embeddings):
    """BAAI/bge-m3 embedding models.

    To use, you should have the ``BAAI/bge-m3`` python package installed, and the
    environment variable ``BAAI/bge-m3_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import BAAI/bge-m3Embeddings
            BAAI/bge-m3 = BAAI/bge-m3Embeddings(
                model="embed-english-light-v2.0", BAAI/bge-m3_api_key="my-api-key"
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
            from FlagEmbedding import BGEM3FlagModel

        except ImportError:
            raise ValueError(
                "Could not import BGEM3FlagModel python package. "
                "Please install it with `pip install FlagEmbedding`."
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to BAAI/bge-m3 embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = embed_model.encode(texts,
                                        batch_size=1,
                                        max_length=8192,
                                        # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                        )['dense_vecs']
        return [list(map(float, e)) for e in embeddings]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to BAAI/bge-m3's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = await embed_model.encode(texts,
                                              batch_size=1,
                                              max_length=8192,
                                              # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                              )['dense_vecs']
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Call out to BAAI/bge-m3's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = embed_model.encode([text],
                                        batch_size=1,
                                        max_length=8192,
                                        # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                        )['dense_vecs']
        print(f"{type(embeddings[0])}")
        return list(map(float, embeddings[0]))

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to BAAI/bge-m3's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = await embed_model.encode([text],
                                              batch_size=1,
                                              max_length=8192,
                                              # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                              )['dense_vecs']
        return list(map(float, embeddings[0]))
