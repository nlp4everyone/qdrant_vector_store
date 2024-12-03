# Return type
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from langchain_core.documents.base import Document
# Qdrant Components
from qdrant_client.models import ScoredPoint
from qdrant_client import models
# Embedding package
from fastembed import SparseTextEmbedding, SparseEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from langchain_core.embeddings import Embeddings
# Other components
from typing import List, Sequence, Literal, Union, Optional
import requests

Embedding = List[float]

class BaseVectorStore:
    def __init__(self,
                 dense_embedding_model: Union[BaseEmbedding, Embeddings],
                 url: str = "http://localhost:6333",
                 sparse_embedding_model: Optional[SparseTextEmbedding] = None,
                 collection_name: str = "qdrant_vector_store",
                 distance: models.Distance = models.Distance.COSINE,
                 shard_number: int = 2,
                 quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
                 default_segment_number: int = 4,
                 on_disk: bool = True,
                 sparse_datatype: models.Datatype = models.Datatype.FLOAT16):
        # Collection name
        self._collection_name = collection_name
        # Optimization params
        self._on_disk = on_disk
        self._distance = distance
        self._shard_number = shard_number
        self._quantization_mode = quantization_mode
        self._default_segment_number = default_segment_number
        # Embedding model
        self._dense_embedding_model = dense_embedding_model
        self._sparse_embedding_model = sparse_embedding_model
        # Datatype
        self._sparse_datatype = sparse_datatype
        # Url
        self._url = url
        # Check health
        if not self.check_health():
            raise Exception(f"Server with url: {self._url} not available!")

    def check_health(self):
        try:
            # Replace with your Qdrant host and port
            health_url = f"{self._url}/collections"

            # Make request to check status
            response = requests.get(health_url)
            # Return status
            return True if response.status_code == 200 else False
        except requests.exceptions.ConnectionError as e:
            # Return status
            return False

    @staticmethod
    def get_quantization_config(quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
                                always_ram :bool = True):
        """
        Get quantization config with mode
        :param quantization_mode: Include scalar, binary and product.
        :param always_ram: Indicated that quantized vectors is persisted on RAM.
        :return:
        """
        # Define quantization mode if enable
        if quantization_mode == "scalar":
            # Scalar mode, currently Qdrant only support INT8
            quantization_config = models.ScalarQuantization(
                scalar = models.ScalarQuantizationConfig(
                    type = models.ScalarType.INT8,
                    quantile = 0.99,
                    # if specify 0.99, 1% of extreme values will be excluded from the quantization bounds.
                    always_ram = always_ram
                )
            )
        elif quantization_mode == "binary":
            # Binary mode
            quantization_config = models.BinaryQuantization(
                binary = models.BinaryQuantizationConfig(
                    always_ram = always_ram,
                ),
            ),
        else:
            # Product quantization mode
            quantization_config = models.ProductQuantization(
                product = models.ProductQuantizationConfig(
                    compression = models.CompressionRatio.X16,  # Default X16
                    always_ram = always_ram,
                ),
            )
        return quantization_config

    @staticmethod
    def embed_sparse_text(contents :List[str],
                          fastembed_model :SparseTextEmbedding,
                          batch_size :int = 32,
                          parallel :int = 1) -> List[SparseEmbedding]:
        """

        :param backends: Select backend for sparse embedding. Default is default (Required fastembed_model model)
        :param fastembed_model: Fastembed sparse embedding model (Optional)
        :param batch_size: Batch size for embed (int)
        :param parallel: Number of parallel processing (int)
        :return:
        """
        # Check backend
        # Fastembed case
        return list(fastembed_model.embed(documents = contents))

    @staticmethod
    def get_dense_embedding_config(embedding_dimension :int,
                                   distance :models.Distance,
                                   on_disk :bool,
                                   dense_embedding_model :Union[BaseEmbedding,Embeddings],
                                   datatype :models.Datatype = models.Datatype.FLOAT16) -> dict:
        # Define vector config
        dense_vectors_config = models.VectorParams(size = embedding_dimension,
                                                   distance = distance,
                                                   on_disk = on_disk,
                                                   hnsw_config = models.HnswConfigDiff(on_disk = on_disk),
                                                   datatype = datatype)

        # Define config with BaseEmbedding
        if isinstance(dense_embedding_model, BaseEmbedding):
            return {dense_embedding_model.model_name : dense_vectors_config}
        # Return config with Embeddings Langchain
        return {dense_embedding_model.model: dense_vectors_config}

    @staticmethod
    def get_sparse_embedding_config(sparse_embedding_model: SparseTextEmbedding,
                                    datatype :models.Datatype) -> dict:
        # Define sparse model name
        sparse_model_name = sparse_embedding_model.model_name
        # Return config
        return {sparse_model_name: models.SparseVectorParams(
            index = models.SparseIndexParams(datatype = datatype)
        )}

    @staticmethod
    def _convert_documents_to_payloads(documents: Union[Sequence[BaseNode], Sequence[Document]]) -> list[dict]:
        """
        Construct the payload data from LlamaIndex document/node datatype

        Args:
            documents (BaseNode): The list of BaseNode datatype in LlamaIndex
        Returns:
            Payloads (list[dict]).
        """
        # LlamaIndex BaseNode case
        if isinstance(documents[0],BaseNode):
            # Define input type
            # documents = Sequence[BaseNode]

            # Clear private data from payload
            for i in range(len(documents)):
                documents[i].embedding = None
                # Pop file path
                documents[i].metadata["file_path"] = "",
                # documents[i].excluded_embed_metadata_keys = []
                # documents[i].excluded_llm_metadata_keys = []
                # Remove metadata in relationship
                for key in documents[i].relationships.keys():
                    documents[i].relationships[key].metadata = {}

            # Get payloads
            payloads = [{"_node_content": document.dict(),
                         "_node_type": document.class_name(),
                         "doc_id": document.id_,
                         "document_id": document.id_,
                         "ref_doc_id": document.id_} for document in documents]
        else:
            # Langchain Document case
            payloads = [{"page_content": document.page_content,
                         "metadata": document.metadata,
                         "_node_type": document.type} for document in documents]
        return payloads

    @staticmethod
    def _convert_score_point_to_node_with_score(scored_points: List[ScoredPoint]) -> Sequence[NodeWithScore]:
        """
        Convert ScorePoint Datatype (Qdrant) to NodeWithScore Datatype (LlamaIndex)

        Args:
            scored_points (List[ScoredPoint]): List of ScoredPoint
        Returns:
            Sequence of NodeWithScore
        """

        # Define text nodes
        text_nodes = [TextNode.from_dict(point.payload["_node_content"]) for point in scored_points]
        # return NodeWithScore
        return [NodeWithScore(node=text_nodes[i], score=point.score) for (i, point) in enumerate(scored_points)]

    @staticmethod
    def _convert_score_point_to_document(scored_points: List[ScoredPoint]) -> Sequence[Document]:
        """
        Convert ScorePoint Datatype (Qdrant) to Document Datatype (LlamaIndex)

        Args:
            scored_points (List[ScoredPoint]): List of ScoredPoint
        Returns:
            Sequence of NodeWithScore
        """

        return [Document(page_content = point.payload["page_content"],
                         metadata = point.payload["metadata"],
                         id = point.id) for point in scored_points]
