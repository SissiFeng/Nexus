import argparse
import sys
from pathlib import Path


def setup() -> None:
    """Interactively set the model API key and save to .env."""
    env_path = Path(__file__).resolve().parent.parent / ".env"

    # Show current status
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("MODEL_API_KEY=") or line.startswith("ANTHROPIC_API_KEY="):
                masked = line.split("=", 1)[1]
                if masked and masked != "your-api-key-here":
                    masked = masked[:8] + "..." + masked[-4:]
                    print(f"Current API key: {masked}")
                break

    key = input("Enter your model API key: ").strip()
    if not key:
        print("No key provided, aborted.")
        sys.exit(1)

    if not key.startswith("sk-ant-"):
        print("Warning: key doesn't start with 'sk-ant-', are you sure? (y/n) ", end="")
        if input().strip().lower() != "y":
            print("Aborted.")
            sys.exit(1)

    env_path.write_text(f"MODEL_API_KEY={key}\n")
    print(f"API key saved to {env_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimization Copilot (Opus 4.6)")
    parser.add_argument("command", nargs="?", default="chat",
                        choices=["chat", "setup", "usage"],
                        help="'setup' config API key, 'usage' show stats, 'chat' start (default)")
    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "usage":
        from .usage import summary
        print(summary())
    else:
        from .agent import run_conversation
        run_conversation()


if __name__ == "__main__":
    main()
