# Qdrant components
from qdrant_client import models, http
# Embedding type
from llama_index.core.base.embeddings.base import BaseEmbedding
from langchain_core.embeddings import Embeddings
# Inheritance
from qdrant_vector_store.base import BaseVectorStore
# Other components
from typing import Optional, Union, List, Literal
# Sparse embedding
from fastembed import SparseTextEmbedding, SparseEmbedding
from uuid import uuid4
import requests

# DataType
Num = Union[int, float]
Embedding = List[float]
# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16


class AsyncBaseVectorStore(BaseVectorStore):
    def __init__(self,
                 dense_embedding_model: Union[BaseEmbedding, Embeddings],
                 sparse_embedding_model: Optional[SparseTextEmbedding] = None,
                 url: str = "http://localhost:6333",
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
        # Client
        self._client = None
        # Check health
        if not self._check_health():
            raise Exception(f"Server with url: {self._url} not available!")

    def _check_health(self):
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

    async def _embed_texts(self,
                           texts: list[str],
                           batch_size: int,
                           num_workers: int,
                           show_progress: bool = True) -> List[Embedding]:
        """
        Return embedding from documents

        Args:
            texts (list[str]): List of input text
            batch_size (int): The desired batch size
            num_workers (int): The desired num workers
            show_progress (bool): Indicate show progress or not

        Returns:
             Return list of Embedding
        """
        # Base Embedding encode text
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # Re-defined model
            self._dense_embedding_model: BaseEmbedding
            # Set batch size and num workers
            self._dense_embedding_model.num_workers = num_workers
            self._dense_embedding_model.embed_batch_size = batch_size

            # Other information
            model_infor = self._dense_embedding_model.dict()
            callback_manager = self._dense_embedding_model.callback_manager
            # Return embedding
            embeddings = await self._dense_embedding_model.aget_text_embedding_batch(texts = texts,
                                                                                     show_progress = show_progress)
        else:
            # Langchain Embedding
            self._dense_embedding_model: Embeddings
            # Return embedding
            embeddings = await self._dense_embedding_model.aembed_documents(texts = texts)
        return embeddings

    async def _insert_points(self,
                             dense_embeddings: list[list[float]],
                             payloads: list[dict],
                             sparse_embeddings: Optional[List[SparseEmbedding]] = None,
                             point_ids: Optional[list[str]] = None,
                             batch_size: int = _DEFAULT_UPLOAD_BATCH_SIZE,
                             parallel: int = 1) -> None:
        """
        Insert point to the collection

        Args:
            list_embeddings (Required): List of embeddings
            list_payloads (List[dict]) (Required): List of payloads
            point_ids (list[str]) (Optional): List of point id
            batch_size (int): The desired batch size
            parallel (int): The desired batch size
        """

        # Check size
        if not len(dense_embeddings) == len(payloads):
            raise Exception("Number of embeddings must be equal with number of payloads")
        # When point not specify
        if point_ids == None: point_ids = [str(uuid4()) for i in range(len(dense_embeddings))]

        # Define model name
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # LlamaIndex Base Embedding with model name
            model_name = self._dense_embedding_model.model_name
        else:
            # Langchain Embeddings with model name
            model_name = self._dense_embedding_model.model

        # Define point
        points = [models.PointStruct(id = point_ids[i],
                                     vector = {model_name: dense_embeddings[i]},
                                     payload = payloads[i]) for i in range(len(dense_embeddings))]
        # Add sparse embedding
        if sparse_embeddings != None:
            # Sparse embedding model name
            sparse_model_name = self._sparse_embedding_model.model_name
            # Add sparse
            for i in range(len(points)):
                points[i].vector.update({sparse_model_name: models.SparseVector(indices = sparse_embeddings[i].indices,
                                                                                values = sparse_embeddings[i].values)})

        # Upload points
        await self._client.upload_points(collection_name = self._collection_name,
                                         points = points,
                                         batch_size = batch_size,
                                         parallel = parallel)

    async def _count_points(self) -> int:
        """Return the total amount of point inside collection"""
        # Check collection exist
        status = await self._client.collection_exists(self._collection_name)
        if not status:
            raise Exception(f"Collection {self._collection_name} is not exist!")

        # Get total amount of points
        result = await self._client.count(self._collection_name)
        return result.count

    async def _collection_info(self) -> http.models.CollectionInfo:
        """Return the total amount of point inside collection"""
        # Check collection exist
        status = await self._client.collection_exists(self._collection_name)
        if not status:
            raise Exception(f"Collection {self._collection_name} is not exist!")

        # Get collection info
        collection_info = await self._client.get_collection(collection_name=self._collection_name)
        return collection_info

    async def _create_collection(self,
                                 dense_vectors_config: Union[models.VectorParams, dict],
                                 sparse_vectors_config: Optional[dict[str, models.SparseVectorParams]] = None,
                                 shard_number: int = 2,
                                 quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
                                 default_segment_number: int = 4,
                                 always_ram: bool = True) -> None:
        """
        Create collection with default name

        Args:
            dense_vectors_config: Config for dense vector
            sparse_vectors_config: Config for sparse vector
            shard_number: The number of parallel processes as the same time. Default is 2.
            quantization_mode: Quantization mode.
            default_segment_number: Default is 4. Larger value will enhance the latency, smaller one the throughput.
            always_ram: Indicated that quantized vectors is persisted on RAM.
        """
        # When collection not existed!
        status = await self._client.collection_exists(self._collection_name)
        if not status:
            quantization_config = self._get_quantization_config(quantization_mode=quantization_mode,
                                                                always_ram=always_ram)

            # Optimizer config
            # When indexing threshold is 0, It will enable to avoid unnecessary indexing of vectors,
            # which will be overwritten by the next batch.
            optimizers_config = models.OptimizersConfigDiff(default_segment_number=default_segment_number,
                                                            indexing_threshold=0)

            # Create collection
            await self._client.create_collection(collection_name=self._collection_name,
                                                 vectors_config=dense_vectors_config,
                                                 sparse_vectors_config=sparse_vectors_config,
                                                 shard_number=shard_number,
                                                 quantization_config=quantization_config,
                                                 optimizers_config=optimizers_config)
            # Update collection
            await self._client.update_collection(collection_name=self._collection_name,
                                                 optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))