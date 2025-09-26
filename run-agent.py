from agents import SimpleOllamaResearchAgent, OllamaResearchAgent
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Wrong number of arguments")

    agent = OllamaResearchAgent(use_llm_filter=True) if sys.argv[1] == "BASE" else SimpleOllamaResearchAgent()
    print("Using agent:", agent)

    user_request = """
    Summarize the latest breakthroughs in quantum error correction
    and superconducting qubits research. Provide references if possible.
    """

    print("[Main] Asking agent...")
    answer = agent.ask(user_request)
    print("\n=== Agent Answer ===\n")
    print(answer)
