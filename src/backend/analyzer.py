def classify_prompt(prompt: str) -> str:
    prompt = prompt.lower()
    if any(x in prompt for x in ["code", "function", "python", "java", "bfs", "sort"]):
        return "coding"
    if any(x in prompt for x in ["math", "calculate", "solve", "equation", "integral"]):
        return "math"
    if any(x in prompt for x in ["why", "explain", "reason", "logic"]):
        return "reasoning"
    if any(x in prompt for x in ["what is", "who", "when", "fact", "capital"]):
        return "qa"
    return "qa"

