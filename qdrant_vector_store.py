from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams,Filter
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Optional, Union, List, Sequence, Literal
from .base_vector_store import BaseVectorStore
from uuid import uuid4
# DataType
Num = Union[int, float]
Embedding = List[float]
# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 16

class QdrantVectorStore(BaseVectorStore):
    def __init__(self,
                 dense_embedding_model: BaseEmbedding,
                 collection_name: str = "qdrant_vector_store",
                 url :str = "http://localhost:6333",
                 port :int = 6333,
                 grpc_port :int = 6334,
                 prefer_grpc :bool = False,
                 api_key :Optional[str] = None,
                 distance: Distance = Distance.COSINE,
                 shard_number: int = 2,
                 quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
                 default_segment_number: int = 4,
                 on_disk: bool = True) -> None:

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
        self._client = QdrantClient(url = url,
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

    def _embed_texts(self,
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
        # Set batch size and num workers
        self.__dense_embedding_model.num_workers = num_workers
        self.__dense_embedding_model.embed_batch_size = batch_size
        # Other information
        model_infor = self.__dense_embedding_model.dict()
        callback_manager = self.__dense_embedding_model.callback_manager
        # Return embedding
        return self.__dense_embedding_model.get_text_embedding_batch(texts = texts,
                                                                     show_progress = show_progress)

    def __create_collection(self,
                            dense_vectors_config: Union[VectorParams,dict],
                            sparse_vectors_config :Optional[str] = None,
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
        if not self._client.collection_exists(self.__collection_name):
            quantization_config = self._get_quantization_config(quantization_mode = quantization_mode,
                                                                always_ram = always_ram)

            # Optimizer config
            # When indexing threshold is 0, It will enable to avoid unnecessary indexing of vectors,
            # which will be overwritten by the next batch.
            optimizers_config = models.OptimizersConfigDiff(default_segment_number = default_segment_number,
                                                            indexing_threshold = 0)

            # Define dense config
            dense_vectors_config = {self.__dense_embedding_model.model_name :dense_vectors_config}

            # Create collection
            self._client.create_collection(
                collection_name = self.__collection_name,
                vectors_config = dense_vectors_config,
                sparse_vectors_config = sparse_vectors_config,
                shard_number = shard_number,
                quantization_config = quantization_config,
                optimizers_config = optimizers_config
            )
            # Update collection
            self._client.update_collection(
                collection_name = self.__collection_name,
                optimizer_config = models.OptimizersConfigDiff(indexing_threshold = 20000),
            )

    def __insert_points(self,
                        list_embeddings :list[list[float]],
                        list_payloads :list[dict],
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
        if not len(list_embeddings) == len(list_payloads):
            raise Exception("Number of embeddings must be equal with number of payloads")
        # When point not specify
        if point_ids == None: point_ids = [str(uuid4()) for i in range(len(list_embeddings))]

        # Define model name
        model_name = self.__dense_embedding_model.model_name
        # Define point
        points = [models.PointStruct(id = point_ids[i],
                                     vector = {model_name: embedding},
                                     payload = list_payloads[i]) for (i, embedding) in enumerate(list_embeddings)]

        # Upload points
        self._client.upload_points(collection_name = self.__collection_name,
                                   points = points,
                                   batch_size = batch_size,
                                   parallel = parallel)

    def insert_documents(self,
                         documents :Sequence[BaseNode],
                         embedded_batch_size: int = 32,
                         embedded_num_workers: Optional[int] = None,
                         upload_batch_size: int = 16,
                         upload_parallel :Optional[int] = 1) -> None:
        """
        Insert document to specified collection.

        :param documents: List of BaseNode.
        :type documents: Sequence[BaseNode]
        :param embedded_batch_size: Batch size for embedding model. Default is 64.
        :type embedded_batch_size: int
        :param embedded_num_workers: Batch size for embedding model (Optional). Default is None.
        :type embedded_num_workers: int
        :param upload_batch_size: Batch size for uploading points. Default is 16.
        :type upload_batch_size: int
        :param upload_parallel: Number of parallel for uploading point (Optional). Default is None.
        :type upload_parallel: Optional[int]
        """
        # Get content and its embedding
        contents = [doc.get_content() for doc in documents]

        # Process text embedding with LlamaIndex Embedding
        embeddings = self._embed_texts(texts = contents,
                                       batch_size = embedded_batch_size,
                                       num_workers = embedded_num_workers)

        # Get embedding dimension
        embedding_dimension = len(embeddings[0])
        # Define vector config
        dense_vectors_config = VectorParams(size = embedding_dimension,
                                            distance = self.__distance,
                                            on_disk = self.__on_disk,
                                            hnsw_config = models.HnswConfigDiff(on_disk = self.__on_disk))

        # Define payloads
        payloads = self._convert_documents_to_payloads(documents = documents)

        # Create collection if doesn't exist!
        self.__create_collection(dense_vectors_config = dense_vectors_config,
                                 shard_number = self.__shard_number,
                                 quantization_mode = self.__quantization_mode,
                                 default_segment_number = self.__default_segment_number)

        # Insert vector to collection with BaseEmbedding model
        self.__insert_points(list_embeddings = embeddings,
                             list_payloads = payloads,
                             batch_size = upload_batch_size,
                             parallel = upload_parallel)

    def __query(self,
                query: str,
                similarity_top_k: int = 3,
                condition_filter: Optional[Filter] = None,
                score_threshold :Optional[float] = None,
                rescore :bool = True,
                return_type :Literal["NodeWithScore","default"] = "NodeWithScore"
                ) -> Union[Sequence[NodeWithScore],List[models.ScoredPoint],List[models.QueryResponse]]:
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
        # Get query embedding with Llama Embedding
        query_embedding = self.__dense_embedding_model.get_query_embedding(query = query)

        # Default value
        search_params = None
        # Disable rescore method
        if not rescore:
            search_params = models.SearchParams(
                quantization = models.QuantizationSearchParams(rescore = False)
            )
        # Model name
        model_name = self.__dense_embedding_model.model_name
        query_vector = (model_name, query_embedding)

        # Return search
        scored_points = self._client.search(collection_name = self.__collection_name,
                                            query_vector = query_vector,
                                            limit = similarity_top_k,
                                            search_params = search_params,
                                            query_filter = condition_filter,
                                            score_threshold = score_threshold)
        # Convert to node with score
        return self._convert_score_point_to_node_with_score(scored_points = scored_points) if return_type == "NodeWithScore" else scored_points

    def retrieve(self,
                 query: str,
                 similarity_top_k: int = 3,
                 condition_filter: Optional[Filter] = None) -> Sequence[NodeWithScore]:
        """
        Retrieve nodes from vector store corresponding to question.

        :param query: The query str for retrieve (Required)
        :type query: str
        :param similarity_top_k: Default is 3. Return top-k element from retrieval.
        :type similarity_top_k: int
        :param condition_filter: Conditional filter for searching. Default is None
        :type condition_filter: Filter
        :return: Return a sequence of NodeWithScore
        :rtype Sequence[NodeWithScore]
        """

        # Check base collection
        if not self._client.collection_exists(self.__collection_name):
            raise ValueError(f"Collection {self.__collection_name} isn't existed")

        # Check collection
        count_points = self._count_points()
        if count_points == 0:
            raise Exception(f"Collection {self.__collection_name} is empty!")

        # Enable base search
        return self.__query(query = query,
                            similarity_top_k = similarity_top_k,
                            condition_filter = condition_filter,
                            return_type = "NodeWithScore")

    def _count_points(self) -> int:
        """Return the total amount of point inside collection"""
        # Check collection exist
        if not self._client.collection_exists(self.__collection_name):
            raise Exception(f"Collection {self.__collection_name} is not exist!")

        # Get total amount of points
        return self._client.count(self.__collection_name).count

