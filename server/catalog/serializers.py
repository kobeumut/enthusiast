from rest_framework import serializers
from utils.serializers import ParentDataContextSerializerMixin

from agent.core.registries.embeddings.embedding_provider_registry import EmbeddingProviderRegistry
from sync.document.registry import DocumentSourcePluginRegistry
from sync.product.registry import ProductSourcePluginRegistry

from .models import (
    EMBEDDING_VECTOR_DIMENSIONS,
    DataSet,
    Document,
    DocumentSource,
    ECommerceIntegration,
    Product,
    ProductSource,
)
from .utils import PydanticModelField


class DataSetSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSet
        fields = [
            "id",
            "name",
            "language_model_provider",
            "language_model",
            "embedding_provider",
            "embedding_model",
            "embedding_vector_dimensions",
        ]

class DataSetCreateSerializer(DataSetSerializer):
    preconfigure_agents = serializers.BooleanField(write_only=True, required=False, default=False)

    class Meta(DataSetSerializer.Meta):
        fields = DataSetSerializer.Meta.fields + ["preconfigure_agents"]

    def validate(self, data):
        """Validate embedding configuration for a new data set.

        Two layers of validation, in order:

        1. The vector dimension must equal the platform-wide ``EMBEDDING_VECTOR_DIMENSIONS``.
           Every data set stores its chunks in the same fixed pgvector column
           (``vector(EMBEDDING_VECTOR_DIMENSIONS)``), so a non-matching dimension would crash
           every chunk insert at runtime. This is an explicit product decision, not a tunable
           setting.
        2. The chosen embedding model must actually support that dimension, per the
           provider's ``vector_size_constraints()``.
        """
        embedding_vector_dimensions = data.get("embedding_vector_dimensions")

        if (
            embedding_vector_dimensions is not None
            and embedding_vector_dimensions != EMBEDDING_VECTOR_DIMENSIONS
        ):
            raise serializers.ValidationError(
                {
                    "embedding_vector_dimensions": (
                        f"Embedding vector dimensions must be {EMBEDDING_VECTOR_DIMENSIONS} \u2014 the "
                        f"fixed dimension of the shared chunk-table column. Got {embedding_vector_dimensions}."
                    )
                }
            )

        embedding_provider = data.get("embedding_provider")
        embedding_model = data.get("embedding_model")
        if embedding_provider and embedding_model and embedding_vector_dimensions is not None:
            try:
                provider_class = EmbeddingProviderRegistry().provider_class_by_name(embedding_provider)
            except Exception:
                provider_class = None

            if provider_class is not None:
                constraints = provider_class.vector_size_constraints()
                allowed_sizes = constraints.get(embedding_model)
                if allowed_sizes and embedding_vector_dimensions not in allowed_sizes:
                    raise serializers.ValidationError(
                        {
                            "embedding_vector_dimensions": (
                                f"Model '{embedding_model}' only supports vector sizes: "
                                f"{allowed_sizes}. Got {embedding_vector_dimensions}."
                            )
                        }
                    )

        return data


class DataSetUpdateSerializer(DataSetSerializer):
    """Serializer for partial updates of a ``DataSet``.

    Embedding configuration (provider, model and vector dimensions) is immutable once a data
    set exists: the chunk tables share a single fixed-dimension pgvector column, so changing
    any of these would silently invalidate the embeddings already stored and mix old chunks
    with new query vectors. An attempt to change them is rejected with a clear error pointing
    at recreating the data set (or a reindex/migration) instead.
    """

    class Meta(DataSetSerializer.Meta):
        read_only_fields = ("embedding_provider", "embedding_model", "embedding_vector_dimensions")

    def validate(self, attrs):
        instance = self.instance
        if instance is not None:
            for field in self.Meta.read_only_fields:
                if field in self.initial_data:
                    incoming = self.initial_data[field]
                    current = getattr(instance, field)
                    # Request data may arrive typed differently than the stored value
                    # (e.g. a JSON number vs. an int), so compare on the string form.
                    if str(incoming) != str(current):
                        raise serializers.ValidationError(
                            {
                                field: (
                                    "Embedding configuration is immutable after a data set is created, "
                                    "because changing it would invalidate the chunk embeddings already "
                                    "stored in the fixed pgvector column. Create a new data set (or "
                                    "perform a reindex/migration) instead."
                                )
                            }
                        )
        return attrs


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ["name", "slug", "sku", "description", "categories", "properties", "price"]


class DocumentSerializer(serializers.ModelSerializer):
    is_indexed = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = ["url", "title", "content", "is_indexed"]

    def get_is_indexed(self, obj):
        return obj.chunks_count > 0


class ProductSourceConfigSerializer(serializers.Serializer):
    configuration_args = PydanticModelField(
        config_field_name="CONFIGURATION_ARGS",
        plugin_registry_class=ProductSourcePluginRegistry,
        allow_null=True,
        default=None,
    )


class DocumentSourceConfigSerializer(serializers.Serializer):
    configuration_args = PydanticModelField(
        config_field_name="CONFIGURATION_ARGS",
        plugin_registry_class=DocumentSourcePluginRegistry,
        allow_null=True,
        default=None,
    )


class ProductSourceSerializer(ParentDataContextSerializerMixin, serializers.ModelSerializer):
    context_keys_to_propagate = ["plugin_name"]

    config = ProductSourceConfigSerializer()
    task_id = serializers.CharField(read_only=True, required=False, allow_null=True)

    class Meta:
        model = ProductSource
        fields = ["id", "plugin_name", "config", "data_set_id", "corrupted", "task_id"]


class DocumentSourceSerializer(ParentDataContextSerializerMixin, serializers.ModelSerializer):
    context_keys_to_propagate = ["plugin_name"]

    config = DocumentSourceConfigSerializer()
    task_id = serializers.CharField(read_only=True, required=False, allow_null=True)

    class Meta:
        model = DocumentSource
        fields = ["id", "plugin_name", "config", "data_set_id", "corrupted", "task_id"]


class ECommerceIntegrationSerializer(ParentDataContextSerializerMixin, serializers.ModelSerializer):
    context_keys_to_propagate = ["plugin_name"]

    task_id = serializers.CharField(read_only=True, required=False, allow_null=True)

    class Meta:
        model = ECommerceIntegration
        fields = ["id", "plugin_name", "config", "data_set_id", "task_id"]


class SyncResponseSerializer(serializers.Serializer):
    task_id = serializers.CharField()
