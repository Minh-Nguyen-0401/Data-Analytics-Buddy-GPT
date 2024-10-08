from langchain_openai import ChatOpenAI


def load_llm(model_name):
    """Load Large Language Model.

    Args:
        model_name (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if model_name == "gpt-3.5-turbo":
        return ChatOpenAI(
            model = model_name,
            temperature = 0.0,
            max_tokens = 1000
        )
    elif model_name == "gpt-4":
        return ChatOpenAI(
            model = model_name,
            temperature = 0.0,
            max_tokens = 1000
        )
    elif model_name == "gpt-4o":
        return ChatOpenAI(
            model = model_name,
            temperature = 0.0,
            max_tokens = 1000
        )
    else:
        raise ValueError(f"Unknown model: {model_name}.\
                            Please choose from [gpt-3.5-turbo, gpt-4, ...].")