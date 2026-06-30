from unittest.mock import MagicMock, patch

import pytest
from django.test import override_settings
from django.urls import reverse
from enthusiast_common.utils import RequiredFieldsModel
from pydantic import Field
from rest_framework import status

from agent.core.registries.agents.agent_registry import AgentRegistry
from agent.core.registries.embeddings.embedding_provider_registry import EmbeddingProviderRegistry
from agent.models import Agent
from catalog.models import DataSet

pytestmark = pytest.mark.django_db


class TestArgs(RequiredFieldsModel):
    test: str = Field(default="")


class MockAgentClass:
    AGENT_KEY = "dummy_agent_type"
    NAME= "Dummy Agent"
    AGENT_ARGS = TestArgs
    PROMPT_INPUT = TestArgs
    PROMPT_EXTENSION = TestArgs
    TOOLS = []
    FILE_UPLOAD = False


@pytest.mark.django_db
class TestDataSetListViewPost:
    MOCK_AGENT_CLASS = MockAgentClass

    @pytest.fixture
    def url(self):
        return reverse("data_set_list")

    @pytest.fixture
    def payload(self):
        return {"name": "New DataSet"}

    @pytest.fixture
    def payload_preconfigure_agents(self):
        return {"name": "New DataSet", "preconfigure_agents": True}

    def test_staff_can_create_dataset(self, admin_api_client, url, payload):
        response = admin_api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_201_CREATED
        assert DataSet.objects.filter(name="New DataSet").exists()

    def test_non_staff_cannot_create_dataset(self, api_client, url, payload):
        response = api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert not DataSet.objects.filter(name="New DataSet").exists()

    def test_staff_dataset_creation_with_no_agent_preconfiguration(self, admin_api_client, url, payload):
        response = admin_api_client.post(url, payload, format="json")

        dataset = DataSet.objects.get(name="New DataSet")
        assert response.status_code == status.HTTP_201_CREATED
        assert not Agent.objects.filter(dataset=dataset).exists()

    @patch.object(AgentRegistry, "get_plugin_classes")
    @override_settings(
        AVAILABLE_AGENTS=['dummy_agent_directory_path.DummyAgent']
    )
    def test_staff_dataset_creation_with_agent_preconfiguration(
        self, mock_agent_registry, admin_api_client, url, payload_preconfigure_agents
    ):
        mock_agent_registry.return_value = [self.MOCK_AGENT_CLASS]
        response = admin_api_client.post(url, payload_preconfigure_agents, format="json")

        dataset = DataSet.objects.get(name="New DataSet")
        assert response.status_code == status.HTTP_201_CREATED
        mock_agent_registry.assert_called()

        assert Agent.objects.filter(dataset=dataset).exists()

    @patch.object(EmbeddingProviderRegistry, "provider_class_by_name")
    def test_dataset_creation_with_constrained_vector_size_success(
        self, mock_provider_class_by_name, admin_api_client, url
    ):
        mock_provider = MagicMock()
        mock_provider.vector_size_constraints.return_value = {"constrained-embed": [512]}
        mock_provider_class_by_name.return_value = mock_provider

        payload = {
            "name": "Constrained DataSet",
            "embedding_provider": "MockProvider",
            "embedding_model": "constrained-embed",
            "embedding_vector_dimensions": 512,
        }
        response = admin_api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_201_CREATED
        assert DataSet.objects.filter(name="Constrained DataSet").exists()

    @patch.object(EmbeddingProviderRegistry, "provider_class_by_name")
    def test_dataset_creation_with_constrained_vector_size_failure(
        self, mock_provider_class_by_name, admin_api_client, url
    ):
        mock_provider = MagicMock()
        mock_provider.vector_size_constraints.return_value = {"constrained-embed": [1024]}
        mock_provider_class_by_name.return_value = mock_provider

        payload = {
            "name": "Constrained DataSet",
            "embedding_provider": "MockProvider",
            "embedding_model": "constrained-embed",
            "embedding_vector_dimensions": 512,
        }
        response = admin_api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert not DataSet.objects.filter(name="Constrained DataSet").exists()

    @pytest.mark.parametrize("dimensions", [256, 1024, 1536, 3072])
    def test_dataset_creation_rejects_non_global_vector_dimensions(self, admin_api_client, url, dimensions):
        payload = {
            "name": "Bad Dimensions DataSet",
            "embedding_provider": "OpenAI",
            "embedding_model": "text-embedding-3-large",
            "embedding_vector_dimensions": dimensions,
        }
        response = admin_api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "embedding_vector_dimensions" in response.json()
        assert not DataSet.objects.filter(name="Bad Dimensions DataSet").exists()

    def test_dataset_creation_defaults_to_global_vector_dimensions(self, admin_api_client, url):
        # Omitting embedding_vector_dimensions must fall back to the global constant (512).
        payload = {"name": "Default Dimensions DataSet"}
        response = admin_api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_201_CREATED
        data_set = DataSet.objects.get(name="Default Dimensions DataSet")
        assert data_set.embedding_vector_dimensions == 512

    @patch.object(EmbeddingProviderRegistry, "provider_class_by_name")
    def test_dataset_creation_evaluates_provider_constraints_against_default_dimension(
        self, mock_provider_class_by_name, admin_api_client, url
    ):
        # When embedding_vector_dimensions is omitted it defaults to the global 512, and the
        # provider constraint check must still run against that effective dimension. A
        # provider/model that does NOT support 512 must be rejected even when the caller did
        # not send a dimension.
        mock_provider = MagicMock()
        mock_provider.vector_size_constraints.return_value = {"constrained-embed": [1024]}
        mock_provider_class_by_name.return_value = mock_provider

        payload = {
            "name": "Constrained Default DataSet",
            "embedding_provider": "MockProvider",
            "embedding_model": "constrained-embed",
        }
        response = admin_api_client.post(url, payload, format="json")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "embedding_vector_dimensions" in response.json()
        assert not DataSet.objects.filter(name="Constrained Default DataSet").exists()
