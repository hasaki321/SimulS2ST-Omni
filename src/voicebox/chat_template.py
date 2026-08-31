def format_chat_prompt_phi3(messages, add_assistant_token=True):
    """
    Convert the messages list into the phi-3 chat template format.

    Args:
        messages: A list of messages containing role and content.

    Returns:
        str: The formatted prompt string.
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Add corresponding tags for system and user messages
        if role in ["system", "user"]:
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        # For assistant messages, add only the start tag if it's the last one
        elif role == "assistant" and msg != messages[-1]:
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        elif role == "assistant" and msg == messages[-1]:
            prompt += f"<|{role}|>\n{content}"

    # If the last message is not from the assistant, add the assistant tag
    if messages[-1]["role"] != "assistant" and add_assistant_token:
        prompt += "<|assistant|>"
    return prompt


def format_chat_prompt_qwen2(messages, add_assistant_token=True):
    """
    Custom function to format chat prompts without tool-related logic.

    Args:
        messages: A list of messages containing role and content.
        add_generation_prompt: Boolean to add a generation prompt at the end.

    Returns:
        str: The formatted prompt string.
    """
    prompt = ""

    if messages[0]["role"] == "system":
        prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
    else:
        prompt += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"

    for message in messages:
        if (
            (message["role"] == "user")
            or (message["role"] == "system" and not message == messages[0])
            or (message["role"] == "assistant" and not message.get("tool_calls"))
        ):
            prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        elif message["role"] == "assistant":
            prompt += f"<|im_start|>{message['role']}"
            if message.get("content"):
                prompt += f"\n{message['content']}"
            prompt += "<|im_end|>\n"

    if add_assistant_token:
        prompt += "<|im_start|>assistant\n"

    return prompt



def gen_chat_prompt_for_tts(text, caption=None):
    if caption is None:
        caption = "Speak the following text as human."
    template = [
        {
            "role": "system",
            "content": "You are a powerful AI assistant for speech generation. You need to speak the provided text following the instruction.",
        },
        {
            "role": "user",
            "content": f"instruction: {caption} \ntext: {text} \n",
        },
    ]
    return format_chat_prompt_phi3(template)
    # return format_chat_prompt_qwen2(template)