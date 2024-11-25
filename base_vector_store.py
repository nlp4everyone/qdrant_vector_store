# Return type
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from langchain_core.documents.base import Document
# Qdrant Components
from qdrant_client.models import ScoredPoint
from qdrant_client import models
# Other components
from typing import List, Sequence, Literal, Union

Embedding = List[float]

class BaseVectorStore:
    @staticmethod
    def _get_quantization_config(
                                 quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
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
