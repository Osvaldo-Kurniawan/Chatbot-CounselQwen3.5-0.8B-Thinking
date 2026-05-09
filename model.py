# """
# model.py
# Loads the HuggingFace model and runs text-generation inference.
# No API key required — runs fully locally.
# """

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from tokenizer import MODEL_NAME, get_tokenizer

_model: AutoModelForCausalLM | None = None


def get_model() -> AutoModelForCausalLM:
    """
    Returns a singleton model instance.
    Loads from MODEL_NAME on first call; returns the cached instance afterwards.
    """
    global _model
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        _model.eval()
    return _model


def _build_prompt(user_input: str, conversation_history: list[dict]) -> str:
    """
    Formats conversation history + new user input into a single prompt string.
    Uses a simple Human/Assistant template compatible with most causal-LM models.
    Adjust the template here to match your custom model's expected format.
    """
    lines = []
    for turn in conversation_history:
        role = "Human" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    lines.append(f"Human: {user_input}")
    lines.append("Assistant:")
    return "\n".join(lines)


def generate_response(
    user_input: str,
    conversation_history: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generates a full response string (non-streaming).

    Args:
        user_input: Latest message from the user.
        conversation_history: List of {"role": "user"|"assistant", "content": str}.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.

    Returns:
        The generated response as a plain string.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    prompt = _build_prompt(user_input, conversation_history)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Stop at the next "Human:" turn if the model keeps generating
    if "Human:" in response:
        response = response.split("Human:")[0].strip()
    return response or "I'm not sure how to respond to that."


def stream_response(
    user_input: str,
    conversation_history: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Streams the response token-by-token using TextIteratorStreamer.

    Yields:
        Text chunks (str) as they are generated.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    prompt = _build_prompt(user_input, conversation_history)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    # Run generation in a background thread so we can yield from the streamer
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for chunk in streamer:
        # Stop streaming if the model starts a new Human turn
        if "Human:" in buffer + chunk:
            break
        buffer += chunk
        yield chunk

    thread.join()

# from __future__ import annotations
# from llama_cpp import Llama

# MODEL_PATH = "tegarganang/CounselQwen3.5-0.8B-Thinking"

# _llm: Llama | None = None

# def get_model() -> Llama:
#     global _llm
#     if _llm is None:
#         _llm = Llama(
#             model_path=MODEL_PATH,
#             n_ctx=2048,          # context window
#             n_gpu_layers=-1,     # offload ALL layers to Metal GPU
#             n_threads=8,         # P-cores on M4
#             verbose=False,
#         )
#     return _llm

# def _build_prompt(user_input: str, history: list[dict]) -> str:
#     # Qwen chat template
#     lines = ["<|im_start|>system\nYou are a helpful counseling assistant.<|im_end|>"]
#     for turn in history:
#         role = turn["role"]
#         lines.append(f"<|im_start|>{role}\n{turn['content']}<|im_end|>")
#     lines.append(f"<|im_start|>user\n{user_input}<|im_end|>")
#     lines.append("<|im_start|>assistant")
#     return "\n".join(lines)

# def generate_response(
#     user_input: str,
#     conversation_history: list[dict],
#     max_new_tokens: int = 256,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
# ) -> str:
#     llm = get_model()
#     prompt = _build_prompt(user_input, conversation_history)
#     result = llm(
#         prompt,
#         max_tokens=max_new_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         stop=["<|im_end|>", "<|im_start|>"],
#         echo=False,
#     )
#     return result["choices"][0]["text"].strip()

# def stream_response(
#     user_input: str,
#     conversation_history: list[dict],
#     max_new_tokens: int = 256,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
# ):
#     llm = get_model()
#     prompt = _build_prompt(user_input, conversation_history)
#     for chunk in llm(
#         prompt,
#         max_tokens=max_new_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         stop=["<|im_end|>", "<|im_start|>"],
#         stream=True,
#         echo=False,
#     ):
#         yield chunk["choices"][0]["text"]