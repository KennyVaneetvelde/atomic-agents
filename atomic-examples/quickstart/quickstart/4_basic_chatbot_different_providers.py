import os
import instructor
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
from dotenv import load_dotenv

load_dotenv()

# Initialize a Rich Console for pretty console outputs
console = Console()

# Memory setup
memory = AgentMemory()

# Initialize memory with an initial message from the assistant
initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
memory.add_message("assistant", initial_message)


# Function to set up the client based on the chosen provider
def setup_client(provider):
    console.log(f"provider: {provider}")
    if provider == "1" or provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=api_key))
        model = "gpt-4o-mini"
    elif provider == "2" or provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = instructor.from_anthropic(Anthropic(api_key=api_key))
        model = "claude-3-5-haiku-20241022"
    elif provider == "3" or provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        model = "mixtral-8x7b-32768"
    elif provider == "4" or provider == "ollama":
        from openai import OpenAI as OllamaClient

        client = instructor.from_openai(OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama"))
        model = "llama3"
    elif provider == "5" or provider == "gemini":
        from openai import OpenAI

        api_key = os.getenv("GEMINI_API_KEY")
        client = instructor.from_openai(
            OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            mode=instructor.Mode.JSON,
        )
        model = "gemini-2.0-flash-exp"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client, model


# Prompt the user to choose a provider from one in the list below.
providers_list = ["openai", "anthropic", "groq", "ollama", "gemini"]
y = "bold yellow"
b = "bold blue"
g = "bold green"
provider_inner_str = (
    f"{' / '.join(f'[[{g}]{i + 1}[/{g}]]. [{b}]{provider}[/{b}]' for i, provider in enumerate(providers_list))}"
)
providers_str = f"[{y}]Choose a provider ({provider_inner_str}): [/{y}]"

provider = console.input(providers_str).lower()

# Set up the client and model based on the chosen provider
client, model = setup_client(provider)

# Agent setup with specified configuration
agent = BaseAgent(config=BaseAgentConfig(client=client, model=model, memory=memory, model_api_parameters={"max_tokens": 2048}))

# Generate the default system prompt for the agent
default_system_prompt = agent.system_prompt_generator.generate_prompt()
# Display the system prompt in a styled panel
console.print(Panel(default_system_prompt, width=console.width, style="bold cyan"), style="bold cyan")

# Display the initial message from the assistant
console.print(Text("Agent:", style="bold green"), end=" ")
console.print(Text(initial_message.chat_message, style="bold green"))

# Start an infinite loop to handle user inputs and agent responses
while True:
    # Prompt the user for input with a styled prompt
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break

    # Process the user's input through the agent and get the response
    input_schema = BaseAgentInputSchema(chat_message=user_input)
    response = agent.run(input_schema)

    agent_message = Text(response.chat_message, style="bold green")
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(agent_message)
