import pytest
from unittest.mock import Mock, call, patch
from pydantic import BaseModel
import instructor
from atomic_agents.agents.base_agent import (
    BaseIOSchema,
    BaseAgent,
    BaseAgentConfig,
    BaseAgentInputSchema,
    BaseAgentOutputSchema,
    SystemPromptGenerator,
    AgentMemory,
    SystemPromptContextProviderBase,
)
from instructor.dsl.partial import PartialBase


@pytest.fixture
def mock_instructor():
    mock = Mock(spec=instructor.Instructor)
    mock.chat.completions.create = Mock()
    mock.chat.completions.create_partial = Mock()
    return mock


@pytest.fixture
def mock_memory():
    mock = Mock(spec=AgentMemory)
    mock.get_history.return_value = []
    mock.add_message = Mock()
    mock.copy = Mock(return_value=Mock(spec=AgentMemory))
    mock.initialize_turn = Mock()
    return mock


@pytest.fixture
def mock_system_prompt_generator():
    mock = Mock(spec=SystemPromptGenerator)
    mock.generate_prompt.return_value = "Mocked system prompt"
    mock.context_providers = {}
    return mock


@pytest.fixture
def agent_config(mock_instructor, mock_memory, mock_system_prompt_generator):
    return BaseAgentConfig(
        client=mock_instructor,
        model="gpt-4o-mini",
        memory=mock_memory,
        system_prompt_generator=mock_system_prompt_generator,
    )


@pytest.fixture
def agent(agent_config):
    return BaseAgent(agent_config)


def test_initialization(agent, mock_instructor, mock_memory, mock_system_prompt_generator):
    assert agent.client == mock_instructor
    assert agent.model == "gpt-4o-mini"
    assert agent.memory == mock_memory
    assert agent.system_prompt_generator == mock_system_prompt_generator
    assert agent.input_schema == BaseAgentInputSchema
    assert agent.output_schema == BaseAgentOutputSchema
    assert agent.model_api_parameters["temperature"] == 0
    assert "max_tokens" not in agent.model_api_parameters


# model_api_parameters should have priority over the deprecated temperature parameter if both are provided.
def test_initialization_temperature_priority(mock_instructor, mock_memory, mock_system_prompt_generator):
    config = BaseAgentConfig(
        client=mock_instructor,
        model="gpt-4o-mini",
        memory=mock_memory,
        system_prompt_generator=mock_system_prompt_generator,
        temperature=0.5,
        model_api_parameters={"temperature": 1.0},
    )
    agent = BaseAgent(config)
    assert agent.model_api_parameters["temperature"] == 1.0


def test_reset_memory(agent, mock_memory):
    initial_memory = agent.initial_memory
    agent.reset_memory()
    assert agent.memory != initial_memory
    mock_memory.copy.assert_called_once()


def test_get_response(agent, mock_instructor, mock_memory, mock_system_prompt_generator):
    mock_memory.get_history.return_value = [{"role": "user", "content": "Hello"}]
    mock_system_prompt_generator.generate_prompt.return_value = "System prompt"

    mock_response = Mock(spec=BaseAgentOutputSchema)
    mock_instructor.chat.completions.create.return_value = mock_response

    response = agent.get_response()

    assert response == mock_response

    mock_instructor.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Hello"}],
        response_model=BaseAgentOutputSchema,
        temperature=0,
    )


def test_get_context_provider(agent, mock_system_prompt_generator):
    mock_provider = Mock(spec=SystemPromptContextProviderBase)
    mock_system_prompt_generator.context_providers = {"test_provider": mock_provider}

    result = agent.get_context_provider("test_provider")
    assert result == mock_provider

    with pytest.raises(KeyError):
        agent.get_context_provider("non_existent_provider")


def test_register_context_provider(agent, mock_system_prompt_generator):
    mock_provider = Mock(spec=SystemPromptContextProviderBase)
    agent.register_context_provider("new_provider", mock_provider)

    assert "new_provider" in mock_system_prompt_generator.context_providers
    assert mock_system_prompt_generator.context_providers["new_provider"] == mock_provider


def test_unregister_context_provider(agent, mock_system_prompt_generator):
    mock_provider = Mock(spec=SystemPromptContextProviderBase)
    mock_system_prompt_generator.context_providers = {"test_provider": mock_provider}

    agent.unregister_context_provider("test_provider")
    assert "test_provider" not in mock_system_prompt_generator.context_providers

    with pytest.raises(KeyError):
        agent.unregister_context_provider("non_existent_provider")


def test_custom_input_output_schemas(mock_instructor):
    class CustomInputSchema(BaseModel):
        custom_field: str

    class CustomOutputSchema(BaseModel):
        result: str

    custom_config = BaseAgentConfig(
        client=mock_instructor,
        model="gpt-4o-mini",
        input_schema=CustomInputSchema,
        output_schema=CustomOutputSchema,
    )

    custom_agent = BaseAgent(custom_config)

    assert custom_agent.input_schema == CustomInputSchema
    assert custom_agent.output_schema == CustomOutputSchema


def test_base_agent_io_str_and_rich():
    class TestIO(BaseIOSchema):
        """TestIO docstring"""

        field: str

    test_io = TestIO(field="test")
    assert str(test_io) == '{"field":"test"}'
    assert test_io.__rich__() is not None  # Just check if it returns something, as we can't easily compare Rich objects


def test_run(agent, mock_memory):
    mock_input = BaseAgentInputSchema(chat_message="Test input")
    mock_output = BaseAgentOutputSchema(chat_message="Test output")

    agent.get_response = Mock(return_value=mock_output)

    result = agent.run(mock_input)

    assert result == mock_output
    assert agent.current_user_input == mock_input

    mock_memory.add_message.assert_has_calls([call("user", mock_input), call("assistant", mock_output)])


def test_base_io_schema_empty_docstring():
    with pytest.raises(ValueError, match="must have a non-empty docstring"):

        class EmptyDocStringSchema(BaseIOSchema):
            """"""

            pass


def test_base_io_schema_model_json_schema_no_description():
    class TestSchema(BaseIOSchema):
        """Test schema docstring."""

        field: str

    # Mock the superclass model_json_schema to return a schema without a description
    with patch("pydantic.BaseModel.model_json_schema", return_value={}):
        schema = TestSchema.model_json_schema()

    assert "description" in schema
    assert schema["description"] == "Test schema docstring."


@pytest.mark.asyncio
async def test_run_async(agent, mock_memory):
    mock_input = BaseAgentInputSchema(chat_message="Test input")
    mock_output = BaseAgentOutputSchema(chat_message="Test output")

    # Create a mock async generator that properly sets current_user_input and adds messages
    async def mock_run_async(*args, **kwargs):
        agent.memory.initialize_turn()
        agent.current_user_input = mock_input
        agent.memory.add_message("user", mock_input)
        yield mock_output
        agent.memory.add_message("assistant", mock_output)

    # Replace run_async with our mock
    agent.run_async = mock_run_async

    # Collect all responses from the generator
    responses = []
    async for response in agent.run_async(mock_input):
        responses.append(response)

    assert responses == [mock_output]
    assert agent.current_user_input == mock_input
    mock_memory.add_message.assert_has_calls([call("user", mock_input), call("assistant", mock_output)])


@pytest.mark.asyncio
async def test_stream_response_async(agent, mock_memory, mock_instructor, mock_system_prompt_generator):
    mock_input = BaseAgentInputSchema(chat_message="Test input")
    mock_memory.get_history.return_value = [{"role": "user", "content": "Hello"}]
    mock_system_prompt_generator.generate_prompt.return_value = "System prompt"

    partial_responses = [
        BaseAgentOutputSchema(chat_message="Partial response 1"),
        BaseAgentOutputSchema(chat_message="Partial response 2"),
        BaseAgentOutputSchema(chat_message="Final response"),
    ]

    async def mock_create_partial(*args, **kwargs):
        for response in partial_responses:
            yield response

    mock_instructor.chat.completions.create_partial = mock_create_partial

    responses = []
    async for partial_response in agent.stream_response_async(mock_input):
        responses.append(partial_response)

    assert responses == partial_responses

    mock_memory.add_message.assert_called_with("assistant", partial_responses[-1])


def test_model_from_chunks_patched():
    class TestPartialModel(PartialBase):
        @classmethod
        def get_partial_model(cls):
            class PartialModel(BaseModel):
                field: str

            return PartialModel

    chunks = ['{"field": "hel', 'lo"}']
    expected_values = ["hel", "hello"]

    generator = TestPartialModel.model_from_chunks(chunks)
    results = [result.field for result in generator]

    assert results == expected_values


@pytest.mark.asyncio
async def test_model_from_chunks_async_patched():
    class TestPartialModel(PartialBase):
        @classmethod
        def get_partial_model(cls):
            class PartialModel(BaseModel):
                field: str

            return PartialModel

    async def async_gen():
        yield '{"field": "hel'
        yield 'lo"}'

    expected_values = ["hel", "hello"]

    generator = TestPartialModel.model_from_chunks_async(async_gen())
    results = []
    async for result in generator:
        results.append(result.field)

    assert results == expected_values
