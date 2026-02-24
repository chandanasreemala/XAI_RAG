# from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
# from app.config import settings
# from typing import Optional
# import os

# HF_TOKEN = settings.HF_TOKEN
# MODEL = settings.HF_MODEL

# def get_generator(model_name: Optional[str]=None):
#     model_name = model_name or MODEL
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
#     device = 0 if os.getenv("CUDA_AVAILABLE") else -1
#     gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
#     return gen

# def generate_answer(generator, prompt: str, max_length: int = 256, temperature: float = 0.0) -> str:
#     out = generator(prompt, max_length=max_length, do_sample=False, temperature=temperature)
#     return out[0].get("generated_text", "")

from typing import Optional, Tuple, Union
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from app.config import settings

HF_TOKEN = settings.HF_TOKEN
MODEL = settings.HF_MODEL


def get_generator(model_name: Optional[str] = None):
    model_name = model_name or MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)

    # Prefer torch CUDA detection; keep env-var fallback for compatibility.
    device = 0 if (torch.cuda.is_available() or os.getenv("CUDA_AVAILABLE")) else -1

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def _sequence_confidence_exp_mean_logprob(
    generator,
    prompt: str,
    max_new_tokens: int,
) -> Optional[float]:
    """
    Compute confidence as exp(mean token log-probability) over generated tokens.
    Returns value in (0, 1], or None if confidence cannot be computed.
    """
    try:
        model = generator.model
        tokenizer = generator.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        scores = output.scores  # list[tensor], one per generated token
        if not scores:
            return None

        sequence = output.sequences[0]
        generated_token_ids = sequence[-len(scores):]

        log_probs = []
        for step_logits, token_id in zip(scores, generated_token_ids):
            token_log_prob = F.log_softmax(step_logits[0], dim=-1)[int(token_id)]
            log_probs.append(token_log_prob.item())

        if not log_probs:
            return None

        return float(math.exp(sum(log_probs) / len(log_probs)))
    except Exception:
        return None


def generate_answer(
    generator,
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.0,
    return_confidence: bool = False,
) -> Union[str, Tuple[str, Optional[float]]]:
    # Keep deterministic behavior (do_sample=False), as in original code.
    output = generator(
        prompt,
        max_length=max_length,
        do_sample=False,
        temperature=temperature,
    )
    text = output[0].get("generated_text", "")

    if not return_confidence:
        return text

    confidence = _sequence_confidence_exp_mean_logprob(
        generator=generator,
        prompt=prompt,
        max_new_tokens=max_length,
    )
    return text, confidence
