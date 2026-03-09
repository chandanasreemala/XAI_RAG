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

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from app.config import settings

HF_TOKEN = settings.HF_TOKEN
MODEL = settings.HF_MODEL


class GeneratorWrapper:
    """
    Thin wrapper that normalises the interface for both encoder-decoder
    (seq2seq, e.g. Flan-T5) and decoder-only (causal LM, e.g. Llama 3.1,
    Qwen3) models so the rest of the codebase needs no changes.
    """

    def __init__(self, pipe, is_causal: bool, model_name: str) -> None:
        self._pipe = pipe
        self.is_causal = is_causal
        self.model_name = model_name

    @property
    def model(self):
        return self._pipe.model

    @property
    def tokenizer(self):
        return self._pipe.tokenizer

    def __call__(
        self,
        prompt: str,
        max_length: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate text.  Always returns [{"generated_text": answer_only}].
        For causal LMs  -> text-generation pipeline, return_full_text=False
        For seq2seq     -> text2text-generation pipeline, max_length cap
        Temperature is only forwarded when do_sample=True to avoid
        transformer warnings / errors with greedy decoding.
        """
        kwargs: Dict[str, Any] = {"do_sample": do_sample}
        if do_sample and temperature != 1.0:
            kwargs["temperature"] = temperature

        if self.is_causal:
            kwargs["max_new_tokens"] = max_length
            kwargs["return_full_text"] = False
            out = self._pipe(prompt, **kwargs)
            return [{"generated_text": out[0]["generated_text"].strip()}]
        else:
            kwargs["max_length"] = max_length
            return self._pipe(prompt, **kwargs)


def get_generator(model_name: Optional[str] = None) -> "GeneratorWrapper":
    """
    Load a model and return a GeneratorWrapper.
    Auto-detects seq2seq vs causal-LM from the HuggingFace config.
    """
    model_name = model_name or MODEL

    # Detect architecture before downloading weights.
    config = AutoConfig.from_pretrained(model_name, token=HF_TOKEN)
    is_causal = not getattr(config, "is_encoder_decoder", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    # Causal tokenizers often lack a pad token; set it to eos_token so that
    # batched inference and confidence computation work without warnings.
    if is_causal and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer torch CUDA detection; keep env-var fallback for compatibility.
    device = 0 if (torch.cuda.is_available() or os.getenv("CUDA_AVAILABLE")) else -1

    if is_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

    return GeneratorWrapper(pipe, is_causal=is_causal, model_name=model_name)


def _sequence_confidence_exp_mean_logprob(
    generator: "GeneratorWrapper",
    prompt: str,
    max_new_tokens: int,
) -> Optional[float]:
    """
    Compute confidence as the average softmax probability over generated tokens.
    Returns value in (0, 1], or None if confidence cannot be computed.

    Works for both seq2seq and causal-LM:
    seq2seq  -- output.sequences[0] is decoder output (decoder-start + tokens)
    causal   -- output.sequences[0] is input + output tokens; taking the last
                len(scores) elements gives exactly the generated tokens.
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

        scores = output.scores  # list[tensor], one per generated token step
        if not scores:
            return None

        # sequence[-len(scores):] gives the newly generated token IDs for
        # both seq2seq (decoder tokens) and causal-LM (post-prompt tokens).
        sequence = output.sequences[0]
        generated_token_ids = sequence[-len(scores):]

        probs = []
        for step_logits, token_id in zip(scores, generated_token_ids):
            token_prob = F.softmax(step_logits[0], dim=-1)[int(token_id)]
            probs.append(token_prob.item())

        if not probs:
            return None

        return float(sum(probs) / len(probs))
    except Exception:
        return None


def generate_answer(
    generator: "GeneratorWrapper",
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.0,
    return_confidence: bool = False,
) -> Union[str, Tuple[str, Optional[float]]]:
    # Keep deterministic behaviour (do_sample=False), as in original code.
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
