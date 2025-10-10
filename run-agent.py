from agents import SimpleOllamaResearchAgent, ResearchAgent
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Wrong number of arguments")

    agent = ResearchAgent(use_llm_filter=True, model=sys.argv[2]) if sys.argv[1] == "BASE" else SimpleOllamaResearchAgent()
    print("Using agent:", agent)

    user_request = """Wie ist die aktuelle Lage im Israel-Palestina-Konflikt?"""

    print("[Main] Asking agent...")
    answer = agent.ask(user_request)
    print("\n=== Agent Answer ===\n")
    print(answer)
