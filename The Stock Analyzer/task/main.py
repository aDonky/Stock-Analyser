import os
from dotenv import load_dotenv
from openai import OpenAI

ASSISTANT_NAME = "stock_analyzer_assistant"
ASSISTANT_INSTRUCTIONS = (
    "You're an experienced stock analyzer assistant tasked with analyzing "
    "and visualizing stock market data."
)

def get_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

    base_url = os.getenv("OPENAI_BASE_URL")

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)

    return OpenAI(api_key=api_key)


def get_or_create_assistant(client: OpenAI):
    assistants = client.beta.assistants.list(limit=20)

    existing = None
    for assistant in assistants.data:
        if assistant.name == ASSISTANT_NAME:
            existing = assistant
            break

    if existing:
        print(
            f"Matching `stock_analyzer_assistant` assistant found, "
            f"using the first matching assistant with ID: {existing.id}"
        )
        return existing

    assistant = client.beta.assistants.create(
        name=ASSISTANT_NAME,
        instructions=ASSISTANT_INSTRUCTIONS,
        model="gpt-4o-mini",
    )

    print(
        "No matching `stock_analyzer_assistant` assistant found, "
        f"creating a new assistant with ID: {assistant.id}"
    )
    return assistant


def create_thread_and_run(client: OpenAI, assistant_id: str):
    thread = client.beta.threads.create()
    print(f"Thread created with ID: {thread.id}")

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Tell me your name and instructions. YOU MUST Provide a DIRECT and SHORT response.",
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    print(f"Run initiated with ID: {run.id}")


def main():
    client = get_client()
    assistant = get_or_create_assistant(client)
    create_thread_and_run(client, assistant.id)


if __name__ == "__main__":
    main()