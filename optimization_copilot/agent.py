from anthropic import Anthropic

from .config import ANTHROPIC_API_KEY, MAX_TOKENS, MODEL
from . import usage

client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """\
You are Optimization Copilot, a general-purpose AI agent. \
You help users analyze problems, generate solutions, and execute tasks efficiently. \
Be concise, evidence-based, and actionable.\
"""


def chat(messages: list[dict], system: str = SYSTEM_PROMPT) -> str:
    """Send messages to Claude and return the response text."""
    if not usage.check_budget():
        return "[Budget exhausted — $500.00 spent. Exiting.]"

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    )

    info = usage.record(response.usage.input_tokens, response.usage.output_tokens)
    print(f"  ⚡ tokens: {info['input_tokens']}in/{info['output_tokens']}out "
          f"| ${info['cost']:.4f} this call "
          f"| ${info['remaining']:.2f} left")

    return response.content[0].text


def run_conversation() -> None:
    """Run an interactive multi-turn conversation."""
    messages: list[dict] = []
    print("Optimization Copilot (Opus 4.6) — type 'exit' to quit")
    print("Commands: /usage — show token stats, /budget N — set budget\n")
    print(usage.summary())
    print()

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            print("\n" + usage.summary())
            break

        if user_input == "/usage":
            print("\n" + usage.summary() + "\n")
            continue

        if user_input.startswith("/budget "):
            try:
                amount = float(user_input.split()[1])
                usage.set_budget(amount)
                print(f"  Budget updated to ${amount:.2f}\n")
            except (IndexError, ValueError):
                print("  Usage: /budget 500\n")
            continue

        if not usage.check_budget():
            print("\n  Budget exhausted! Use /budget N to add more.\n")
            break

        messages.append({"role": "user", "content": user_input})
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nCopilot: {reply}\n")
