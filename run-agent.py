from agents import SimpleOllamaResearchAgent, OllamaResearchAgent
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Wrong number of arguments")

    agent = OllamaResearchAgent() if sys.argv[1] == "BASE" else SimpleOllamaResearchAgent()

    user_request = """
    Summarize the latest breakthroughs in quantum error correction
    and superconducting qubits research. Provide references if possible.
    """

    print("[Main] Asking agent...")
    answer = agent.ask(user_request)
    print("\n=== Agent Answer ===\n")
    print(answer)
