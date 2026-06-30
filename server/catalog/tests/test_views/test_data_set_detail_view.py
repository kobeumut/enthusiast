import pytest
from django.urls import reverse
from model_bakery import baker
from rest_framework import status

from catalog.models import EMBEDDING_VECTOR_DIMENSIONS, DataSet

pytestmark = pytest.mark.django_db


@pytest.fixture
def data_set_with_global_dimensions():
    return baker.make(
        DataSet,
        embedding_vector_dimensions=EMBEDDING_VECTOR_DIMENSIONS,
        embedding_provider="OpenAI",
        embedding_model="text-embedding-3-large",
    )


@pytest.mark.django_db
class TestDataSetDetailViewPatch:
    @pytest.fixture
    def url(self, data_set_with_global_dimensions):
        return reverse("data_set_detail", kwargs={"data_set_id": data_set_with_global_dimensions.id})

    def test_can_update_non_embedding_fields(self, admin_api_client, url, data_set_with_global_dimensions):
        response = admin_api_client.patch(
            url, {"name": "Renamed DataSet", "language_model": "gpt-4o-mini"}, format="json"
        )

        assert response.status_code == status.HTTP_200_OK
        data_set_with_global_dimensions.refresh_from_db()
        assert data_set_with_global_dimensions.name == "Renamed DataSet"

    def test_rejects_changing_embedding_vector_dimensions(self, admin_api_client, url, data_set_with_global_dimensions):
        response = admin_api_client.patch(
            url, {"embedding_vector_dimensions": 1024}, format="json"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "embedding_vector_dimensions" in response.json()
        data_set_with_global_dimensions.refresh_from_db()
        assert data_set_with_global_dimensions.embedding_vector_dimensions == EMBEDDING_VECTOR_DIMENSIONS

    def test_rejects_changing_embedding_model(self, admin_api_client, url, data_set_with_global_dimensions):
        original_model = data_set_with_global_dimensions.embedding_model
        response = admin_api_client.patch(
            url, {"embedding_model": "text-embedding-3-small"}, format="json"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "embedding_model" in response.json()
        data_set_with_global_dimensions.refresh_from_db()
        assert data_set_with_global_dimensions.embedding_model == original_model

    def test_rejects_changing_embedding_provider(self, admin_api_client, url, data_set_with_global_dimensions):
        original_provider = data_set_with_global_dimensions.embedding_provider
        response = admin_api_client.patch(
            url, {"embedding_provider": "Mistral"}, format="json"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "embedding_provider" in response.json()
        data_set_with_global_dimensions.refresh_from_db()
        assert data_set_with_global_dimensions.embedding_provider == original_provider

    def test_allows_unchanged_embedding_fields(self, admin_api_client, url, data_set_with_global_dimensions):
        # Sending the same values back must not be treated as a change.
        response = admin_api_client.patch(
            url,
            {
                "embedding_provider": data_set_with_global_dimensions.embedding_provider,
                "embedding_model": data_set_with_global_dimensions.embedding_model,
                "embedding_vector_dimensions": data_set_with_global_dimensions.embedding_vector_dimensions,
                "name": "Still Works",
            },
            format="json",
        )

        assert response.status_code == status.HTTP_200_OK
        data_set_with_global_dimensions.refresh_from_db()
        assert data_set_with_global_dimensions.name == "Still Works"
        assert data_set_with_global_dimensions.embedding_vector_dimensions == EMBEDDING_VECTOR_DIMENSIONS
