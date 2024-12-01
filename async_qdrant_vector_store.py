# Qdrant components
from qdrant_client import AsyncQdrantClient, models, http
# Document type
from llama_index.core.schema import BaseNode, NodeWithScore
from langchain_core.documents.base import Document
# Embedding type
from llama_index.core.base.embeddings.base import BaseEmbedding
from langchain_core.embeddings import Embeddings
# Inheritance
from .base_vector_store import BaseVectorStore
# Other components
from typing import Optional, Union, List, Sequence, Literal
from uuid import uuid4
# Sparse embedding
from fastembed import SparseTextEmbedding, SparseEmbedding
# DataType
Num = Union[int, float]
Embedding = List[float]
# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16

class AsyncQdrantVectorStore(BaseVectorStore):
    def __init__(self,
                 dense_embedding_model :Union[BaseEmbedding,Embeddings],
                 sparse_embedding_model :Optional[SparseTextEmbedding] = None,
                 collection_name: str = "qdrant_vector_store",
                 url :str = "http://localhost:6333",
                 port :int = 6333,
                 grpc_port :int = 6334,
                 prefer_grpc :bool = False,
                 api_key :Optional[str] = None,
                 distance: models.Distance = models.Distance.COSINE,
                 shard_number: int = 2,
                 quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
                 default_segment_number: int = 4,
                 on_disk: bool = True,
                 sparse_datatype :models.Datatype = models.Datatype.FLOAT16) -> None:

        """
        Init Qdrant Vector Service:

        :param collection_name: The name of collection (Required).
        :type collection_name: str
        :param dense_embedding_model: Embedding model for contextual text(Required).
        :type dense_embedding_model: BaseEmbedding
        :param url: The Qdrant url string.
        :type url: str
        :param port: Qdrant port. Default is 6333.
        :type port: int
        :param grpc_port: Grpc port. Default is 6334.
        :type grpc_port: int
        :param prefer_grpc: Whether prefer grpc or not
        :type prefer_grpc: bool
        :param api_key: api key for connecting
        :type api_key: str
        :param distance: The calculated distance for similarity search. Default is Cosine.
        :type distance: Distance
        :param shard_number: The number of parallel processes as the same time. Default is 2.
        :type shard_number: int
        :param quantization_mode: Include scalar, binary and product.
        :type quantization_mode: Literal
        :param default_segment_number: Default is 4. Larger value will enhance the latency, smaller one the throughput.
        :type default_segment_number: int
        """
        assert collection_name, "Collection name must be string"
        self._client = AsyncQdrantClient(url = url,
                                         port = port,
                                         grpc_port = grpc_port,
                                         api_key = api_key,
                                         prefer_grpc = prefer_grpc)
        # Collection name
        self.__collection_name = collection_name
        # Optimization params
        self.__on_disk = on_disk
        self.__distance = distance
        self.__shard_number = shard_number
        self.__quantization_mode = quantization_mode
        self.__default_segment_number = default_segment_number

        # Embedding model
        self.__dense_embedding_model = dense_embedding_model
        self.__sparse_embedding_model = sparse_embedding_model
        # Datatype
        self._sparse_datatype = sparse_datatype

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
        if isinstance(self.__dense_embedding_model,BaseEmbedding):
            # Re-defined model
            self.__dense_embedding_model :BaseEmbedding
            # Set batch size and num workers
            self.__dense_embedding_model.num_workers = num_workers
            self.__dense_embedding_model.embed_batch_size = batch_size

            # Other information
            model_infor = self.__dense_embedding_model.dict()
            callback_manager = self.__dense_embedding_model.callback_manager
            # Return embedding
            embeddings = await self.__dense_embedding_model.aget_text_embedding_batch(texts = texts,
                                                                                      show_progress = show_progress)
        else:
            # Langchain Embedding
            self.__dense_embedding_model :Embeddings
            # Return embedding
            embeddings = await self.__dense_embedding_model.aembed_documents(texts = texts)
        return embeddings

    async def __create_collection(self,
                                  dense_vectors_config: Union[models.VectorParams,dict],
                                  sparse_vectors_config :Optional[dict[str,models.SparseVectorParams]] = None,
                                  shard_number :int = 2,
                                  quantization_mode :Literal['binary','scalar','product','none'] = "scalar",
                                  default_segment_number :int = 4,
                                  always_ram :bool = True) -> None:
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
        status = await self._client.collection_exists(self.__collection_name)
        if not status:
            quantization_config = self.get_quantization_config(quantization_mode = quantization_mode,
                                                               always_ram = always_ram)

            # Optimizer config
            # When indexing threshold is 0, It will enable to avoid unnecessary indexing of vectors,
            # which will be overwritten by the next batch.
            optimizers_config = models.OptimizersConfigDiff(default_segment_number = default_segment_number,
                                                            indexing_threshold = 0)

            # Create collection
            await self._client.create_collection(collection_name = self.__collection_name,
                                                 vectors_config = dense_vectors_config,
                                                 sparse_vectors_config = sparse_vectors_config,
                                                 shard_number = shard_number,
                                                 quantization_config = quantization_config,
                                                 optimizers_config = optimizers_config)
            # Update collection
            await self._client.update_collection(collection_name = self.__collection_name,
                                                 optimizer_config = models.OptimizersConfigDiff(indexing_threshold = 20000))

    async def __insert_points(self,
                              dense_embeddings :list[list[float]],
                              payloads :list[dict],
                              sparse_embeddings: Optional[List[SparseEmbedding]] = None,
                              point_ids: Optional[list[str]] = None,
                              batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                              parallel :int = 1) -> None:
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
        if isinstance(self.__dense_embedding_model, BaseEmbedding):
            # LlamaIndex Base Embedding with model name
            model_name = self.__dense_embedding_model.model_name
        else:
            # Langchain Embeddings with model name
            model_name = self.__dense_embedding_model.model

        # Define point
        points = [models.PointStruct(id = point_ids[i],
                                     vector = {model_name: dense_embeddings[i]},
                                     payload = payloads[i]) for i in range(len(dense_embeddings))]
        # Add sparse embedding
        if sparse_embeddings != None:
            # Sparse embedding model name
            sparse_model_name = self.__sparse_embedding_model.model_name
            # Add sparse
            for i in range(len(points)):
                points[i].vector.update({sparse_model_name : models.SparseVector(indices = sparse_embeddings[i].indices,
                                                                                 values = sparse_embeddings[i].values)})

        # Upload points
        await self._client.upload_points(collection_name = self.__collection_name,
                                         points = points,
                                         batch_size = batch_size,
                                         parallel = parallel)

    async def insert_documents(self,
                               documents :Union[Sequence[BaseNode],Sequence[Document]],
                               embedded_batch_size: int = 32,
                               embedded_num_workers: Optional[int] = None,
                               upload_batch_size: int = 16,
                               upload_parallel :Optional[int] = 1) -> None:
        """
        Insert document to specified collection.

        :param documents: A sequence of BaseNode( LlamaIndex) or Document(Langchain) for uploading.
        :type documents: Union[Sequence[BaseNode],Sequence[Document]]
        :param embedded_batch_size: Batch size for embedding model. Default is 64.
        :type embedded_batch_size: int
        :param embedded_num_workers: Batch size for embedding model (Optional). Default is None.
        :type embedded_num_workers: int
        :param upload_batch_size: Batch size for uploading points. Default is 16.
        :type upload_batch_size: int
        :param upload_parallel: Number of parallel for uploading point (Optional). Default is None.
        :type upload_parallel: Optional[int]
        """
        # Check length of document
        if len(documents) == 0:
            assert "Documents is empty"

        # Get content
        if isinstance(documents[0],BaseNode):
            # LlamaIndex BaseNode case
            contents = [doc.get_content() for doc in documents]
        else:
            # Langchain Document case
            contents = [doc.page_content for doc in documents]

        # Embed vector for each content
        embeddings = await self._embed_texts(texts = contents,
                                             batch_size = embedded_batch_size,
                                             num_workers = embedded_num_workers)
        # Get embedding dimension
        embedding_dimension = len(embeddings[0])

        # Dense config
        dense_vectors_config = self.get_dense_embedding_config(embedding_dimension = embedding_dimension,
                                                               distance = self.__distance,
                                                               on_disk = self.__on_disk,
                                                               dense_embedding_model = self.__dense_embedding_model)

        # Define payloads
        payloads = self._convert_documents_to_payloads(documents = documents)

        # Sparse config
        sparse_vectors_config = None
        sparse_embeddings = None
        if isinstance(self.__sparse_embedding_model, SparseTextEmbedding):
            # Get config
            sparse_vectors_config = self.get_sparse_embedding_config(sparse_embedding_model = self.__sparse_embedding_model,
                                                                     datatype = self._sparse_datatype)
            # Get sparse embedding
            sparse_embeddings = self.embed_sparse_text(contents = contents,
                                                       fastembed_model = self.__sparse_embedding_model)

        # Create collection if doesn't exist!
        await self.__create_collection(dense_vectors_config = dense_vectors_config,
                                       sparse_vectors_config = sparse_vectors_config,
                                       shard_number = self.__shard_number,
                                       quantization_mode = self.__quantization_mode,
                                       default_segment_number = self.__default_segment_number)

        # Insert vector to collection with BaseEmbedding model
        await self.__insert_points(dense_embeddings = embeddings,
                                   sparse_embeddings = sparse_embeddings,
                                   payloads = payloads,
                                   batch_size = upload_batch_size,
                                   parallel = upload_parallel)

    async def retrieve(self,
                       query: str,
                       similarity_top_k: int = 3,
                       query_filter: Optional[models.Filter] = None,
                       score_threshold: Optional[float] = None,
                       rescore: bool = True,
                       mode :Literal["dense","sparse"] = "dense",
                       return_type: Literal["ScoredPoints","auto"] = "auto"
                       ) -> Union[Sequence[models.ScoredPoint],Sequence[Document],Sequence[NodeWithScore]]:
        """
        Retrieve nodes from vector store corresponding to question.

        :param query: The query str for retrieve (Required)
        :type query: str
        :param similarity_top_k: Default is 3. Return top-k element from retrieval.
        :type similarity_top_k: int
        :param condition_filter: Conditional filter for searching. Default is None
        :type condition_filter: Filter
        :param rescore: Specify rescore model or not
        :type rescore: bool
        :return: Return a sequence of NodeWithScore
        :rtype Sequence[NodeWithScore]
        """

        # Check base collection
        status = await self._client.collection_exists(self.__collection_name)
        if not status:
            raise ValueError(f"Collection {self.__collection_name} isn't existed")

        # Check collection
        count_points = await self._count_points()
        if count_points == 0:
            raise Exception(f"Collection {self.__collection_name} is empty!")

        # Check config
        config = await self._collection_info()

        # Wrong config
        if config.config.params.sparse_vectors == None and mode == "sparse":
            raise ValueError(f"Sparse mode not config for {self.__collection_name} collection")
        if config.config.params.vectors == None and mode == "dense":
            raise ValueError(f"Dense mode not config for {self.__collection_name} collection")

        # Switch mode
        if mode == "dense":
            # Get query embedding
            if isinstance(self.__dense_embedding_model, BaseEmbedding):
                # Query embeddings
                embeddings = await self.__dense_embedding_model.aget_query_embedding(query = query)
                # Model name
                model_name = self.__dense_embedding_model.model_name
            else:
                # Query embeddings
                embeddings = await self.__dense_embedding_model.aembed_query(text = query)
                # Model name
                model_name = self.__dense_embedding_model.model
        else:
            # Sparse mode
            query_embedding = list(self.__sparse_embedding_model.query_embed(query = query))
            # Embeddings
            embeddings = models.SparseVector(indices = query_embedding[0].indices,
                                             values = query_embedding[0].values)
            # Model name
            model_name = self.__sparse_embedding_model.model_name

        # Default value
        search_params = None
        # Disable rescore method
        if not rescore:
            search_params = models.SearchParams(
                quantization = models.QuantizationSearchParams(rescore = False)
            )

        # Get result from retrieving
        scored_points = await self._client.query_points(collection_name = self.__collection_name,
                                                        query = embeddings,
                                                        using = model_name,
                                                        limit = similarity_top_k,
                                                        query_filter = query_filter)
        # Return Scored Points
        if return_type == "ScoredPoints":
            return scored_points.points
        # Auto mode
        node_type = scored_points.points[0].payload["_node_type"]
        if node_type == "Document":
            # Langchain Document case
            return self._convert_score_point_to_document(scored_points = scored_points.points)
        # LlamaIndex NodeWithScore case
        return self._convert_score_point_to_node_with_score(scored_points = scored_points.points)

    async def _count_points(self) -> int:
        """Return the total amount of point inside collection"""
        # Check collection exist
        status = await self._client.collection_exists(self.__collection_name)
        if not status:
            raise Exception(f"Collection {self.__collection_name} is not exist!")

        # Get total amount of points
        result = await self._client.count(self.__collection_name)
        return result.count

    async def _collection_info(self) -> http.models.CollectionInfo:
        """Return the total amount of point inside collection"""
        # Check collection exist
        status = await self._client.collection_exists(self.__collection_name)
        if not status:
            raise Exception(f"Collection {self.__collection_name} is not exist!")

        # Get collection info
        collection_info = await self._client.get_collection(collection_name = self.__collection_name)
        return collection_info

