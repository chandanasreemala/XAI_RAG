from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import re

logger = logging.getLogger(__name__)

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


def _safe_prompt(tokenizer, prompt: str, max_new_tokens: int) -> str:
    """
    Truncate `prompt` so that  len(input_tokens) + max_new_tokens  fits within
    the model's maximum sequence length.

    Keeps the LAST `budget` tokens so the question (always at the end of the
    prompt) is preserved when the context must be cut.

    Some HuggingFace tokenizers set model_max_length to sys.maxsize; we cap at
    4096 in that case so we never build an absurdly large input tensor.
    """
    limit = getattr(tokenizer, "model_max_length", 512)
    if limit > 1_000_000:  # guard against tokenizers that set this to sys.maxsize
        limit = 4096
    budget = max(64, limit - max_new_tokens - 16)  # leave headroom for special tokens
    token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(token_ids) <= budget:
        return prompt
    # Preserve the end of the prompt (question + "Answer:") by keeping tail tokens
    return tokenizer.decode(token_ids[-budget:], skip_special_tokens=True)


class GeneratorWrapper:
    """
    Thin wrapper that normalises the interface for both encoder-decoder
    (seq2seq, e.g. Gemma, Flan-T5) and decoder-only (causal LM, e.g. Llama,
    Qwen) models so the rest of the codebase needs no changes.

    Key features:
    - Auto-uses the HuggingFace chat template for instruct causal models so
      answers are constrained to 1-5 words via a system prompt.
    - Strips Qwen3 <think>...</think> chain-of-thought blocks automatically.
    - Stops generation at the first newline for causal models to prevent
      multi-paragraph explanations.
    """

    def __init__(self, pipe, is_causal: bool, model_name: str) -> None:
        self._pipe = pipe
        self.is_causal = is_causal
        self.model_name = model_name
        # True when the tokenizer has a chat template (instruct / chat models)
        self._has_chat_template: bool = (
            is_causal
            and hasattr(pipe.tokenizer, "apply_chat_template")
            and pipe.tokenizer.chat_template is not None
        )

    @property
    def model(self):
        return self._pipe.model

    @property
    def tokenizer(self):
        return self._pipe.tokenizer

    # ── Answer post-processing (causal models only) ──────────────────────────
    @staticmethod
    def _extract_answer(text: str) -> str:
        """
        Post-process raw causal-LM output into a short, clean answer.

        Steps:
        1. Strip Qwen3-style <think>...</think> reasoning blocks.
        2. Remove any leading role/label prefixes ("assistant", "Answer:").
        3. Return only the first non-empty line so we never get multi-
           paragraph explanations.
        """
        # 1. Remove chain-of-thought blocks (Qwen3 thinking mode)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # 2. Drop common prefixes that instruct models sometimes emit
        text = re.sub(
            r"(?i)^(answer\s*:\s*|assistant\s*:\s*|\bassistant\b\s*)", "", text.strip()
        )
        # 3. First non-empty line only
        for line in text.split("\n"):
            line = line.strip().rstrip(".")
            if line:
                return line
        return text.strip()

    # ── Prompt building ───────────────────────────────────────────────────
    def make_qa_prompt(self, context: str, question: str) -> str:
        return make_qa_prompt(self, context, question)

    # ── Generation ────────────────────────────────────────────────────────────
    def __call__(
        self,
        prompt: str,
        max_length: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate text.  Always returns [{"generated_text": answer_only}].
        • For causal LMs  → text-generation pipeline, return_full_text=False
        • For seq2seq     → text2text-generation pipeline, max_length cap
        Temperature is only forwarded when do_sample=True to avoid
        transformer warnings / errors with greedy decoding.
        """
        # Silently truncate the prompt to prevent token-overflow warnings /
        # indexing errors for models with a hard token limit (e.g. 512 for
        # enc-dec models, 4096 for some causal LMs).
        prompt = _safe_prompt(self.tokenizer, prompt, max_length)

        kwargs: Dict[str, Any] = {"do_sample": do_sample}
        if do_sample and temperature != 1.0:
            kwargs["temperature"] = temperature

        if self.is_causal:
            kwargs["max_new_tokens"] = max_length
            kwargs["return_full_text"] = False
            out = self._pipe(prompt, **kwargs)
            raw = out[0]["generated_text"].strip()
            return [{"generated_text": self._extract_answer(raw)}]
        else:
            kwargs["max_length"] = max_length
            return self._pipe(prompt, **kwargs)


def get_generator(model_name: Optional[str] = None) -> "GeneratorWrapper":
    """
    Load a model and return a GeneratorWrapper.
    Auto-detects seq2seq vs causal-LM from the HuggingFace config.

    For causal models we always use device_map="auto" so PyTorch spreads
    weights across available GPUs / CPU when a single GPU is short on memory.
    Models larger than ~7B parameters are additionally loaded in 4-bit
    (bitsandbytes) when the package is available, halving VRAM requirements.
    """
    model_name = model_name or MODEL

    # Detect architecture before downloading weights.
    config = AutoConfig.from_pretrained(
        model_name, token=HF_TOKEN, trust_remote_code=True
    )
    is_causal = not getattr(config, "is_encoder_decoder", False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=HF_TOKEN, trust_remote_code=True
    )
    if is_causal and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = 0 if (torch.cuda.is_available() or os.getenv("CUDA_AVAILABLE")) else -1

    if is_causal:
        # Estimate parameter count to decide whether to quantise
        num_params = getattr(config, "num_parameters", None)
        if num_params is None:
            # rough heuristic from hidden size + num layers
            h = getattr(config, "hidden_size", 0)
            l = getattr(config, "num_hidden_layers", 0)
            num_params = h * l * 12  # very rough transformer estimate
        large_model = num_params > 4_000_000_000  # >4B params

        load_kwargs: dict = {
            "token": HF_TOKEN,
            "trust_remote_code": True,
        }

        if device >= 0:
            # Try 4-bit quantisation for large models (needs bitsandbytes)
            if large_model:
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True
                    )
                    load_kwargs["device_map"] = "auto"
                    logger.info("Loading %s with 4-bit quantisation.", model_name)
                except Exception:
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["torch_dtype"] = torch.float16
                    logger.info(
                        "Loading %s with device_map=auto fp16 (bitsandbytes unavailable).",
                        model_name,
                    )
            else:
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        # When device_map is used the pipeline must not specify device=
        pipe_device = None if "device_map" in load_kwargs else device
        pipe_kwargs = {"model": model, "tokenizer": tokenizer}
        if pipe_device is not None:
            pipe_kwargs["device"] = pipe_device
        pipe = pipeline("text-generation", **pipe_kwargs)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, token=HF_TOKEN, trust_remote_code=True
        )
        pipe = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer, device=device
        )

    return GeneratorWrapper(pipe, is_causal=is_causal, model_name=model_name)


_SYSTEM_INSTRUCTION = (
    "You are a precise question-answering assistant. "
    "Answer the question using only the provided context. "
    "Give the shortest possible direct answer — "
    "ideally 1-5 words, never more than one sentence."
)


def make_qa_prompt(gen: "GeneratorWrapper", context: str, question: str) -> str:
    """
    Build the right prompt format for the loaded model.

    Tries system+user chat template first; falls back to merged user message
    for models that reject a system role (e.g. Gemma); then raw QA format.
    """
    tokenizer = gen.tokenizer
    is_causal = getattr(gen, "is_causal", False)
    has_chat = (
        is_causal
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
    )
    if has_chat:
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}",
                },
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{_SYSTEM_INSTRUCTION}\n\n"
                        f"Context: {context}\n\nQuestion: {question}"
                    ),
                }
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            logger.warning(
                "make_qa_prompt: apply_chat_template failed (%s), using raw format.",
                exc,
            )
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def _sequence_confidence_mean_token_prob(
    generator: "GeneratorWrapper",
    prompt: str,
    max_new_tokens: int,
) -> Optional[float]:
    """
    Compute confidence as the mean softmax probability of generated tokens.
    Returns value in (0, 1], or None if confidence cannot be computed.

    Works for both seq2seq and causal-LM:
    • seq2seq  — output.sequences[0] is decoder output (decoder-start + tokens)
    • causal   — output.sequences[0] is input + output tokens; taking the last
                 len(scores) elements gives exactly the generated tokens.
    """
    try:
        model = generator.model
        tokenizer = generator.tokenizer

        # Truncate to model limit before tokenising (prevents overflow warnings)
        prompt = _safe_prompt(tokenizer, prompt, max_new_tokens)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 512),
        )
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
        generated_token_ids = sequence[-len(scores) :]

        probs = []
        for step_logits, token_id in zip(scores, generated_token_ids):
            # token_log_prob = F.log_softmax(step_logits[0], dim=-1)[int(token_id)]
            # log_probs.append(token_log_prob.item())
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

    confidence = _sequence_confidence_mean_token_prob(
        generator=generator,
        prompt=prompt,
        max_new_tokens=max_length,
    )
    return text, confidence
