"""
Templates for different types of model interactions.
These templates are designed to provide clear, structured prompts for various tasks.
"""

TASK_TEMPLATES = {
    "coding": {
        "template": """You are a programming expert. Write a function to solve the following problem:

{prompt_with_context}

Provide the solution in a clear, well-formatted code block with documentation, type hints, and example usage.""",
        "expected_format": "code"
    },
    
    "math": {
        "template": """You are a mathematics expert. Solve the following problem:

{prompt_with_context}""",
        "expected_format": "text"
    },
    
    "reasoning": {
        "template": """You are a logical reasoning expert. Analyze the following question:

{prompt_with_context}

Format your response as:
Analysis:
[Your detailed analysis]

Conclusion:
[Your final conclusion]""",
        "expected_format": "text"
    },
    
    "qa": {
        "template": """You are a knowledgeable expert. Answer the following question:

{prompt_with_context}

Format your response as:
Answer:
[Your direct answer]

Context:
[Relevant background information]""",
        "expected_format": "text"
    },
    
    "summarization": {
        "template": """You are a summarization expert. Summarize the following text:

{prompt_with_context}

Format your response as:
Summary:
[Your concise summary]

Key Points:
- [Point 1]
- [Point 2]
- [Point 3]""",
        "expected_format": "text"
    },
    
    "translation": {
        "template": """You are a translation expert. Translate the following text:

{prompt_with_context}

Format your response as:
Translation:
[Your translation]

Notes:
[Any important context or explanations]""",
        "expected_format": "text"
    }
}

def get_template(task_type: str) -> str:
    """
    Get the template for a specific task type.
    
    Args:
        task_type (str): The type of task (e.g., 'coding', 'math', 'reasoning', etc.)
        
    Returns:
        str: The template string for the specified task type
    """
    if task_type not in TASK_TEMPLATES:
        return "{prompt_with_context}"
    return TASK_TEMPLATES[task_type]["template"]

def get_expected_format(task_type: str) -> str:
    """
    Get the expected format for a specific task type.
    
    Args:
        task_type (str): The type of task
        
    Returns:
        str: The expected format ('code' or 'text')
    """
    if task_type not in TASK_TEMPLATES:
        return "text"
    return TASK_TEMPLATES[task_type]["expected_format"] 