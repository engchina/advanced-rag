"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.

**Class hierarchy:**

.. code-block::

    VectorStore --> <name>  # Examples: Annoy, FAISS, Milvus

    BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: VespaRetriever

**Main helpers:**

.. code-block::

    Embeddings, Document
"""  # noqa: E501

from typing import Any


def _import_oracleaivector() -> Any:
    from mylangchain.vectorstores.oracleaivector import OracleAIVector

    return OracleAIVector


def __getattr__(name: str) -> Any:
    if name == "OracleAIVector":
        return _import_oracleaivector()
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "OracleAIVector",
]
