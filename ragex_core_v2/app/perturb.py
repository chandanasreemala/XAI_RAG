from typing import List
import random
import spacy
import nltk
from nltk.corpus import wordnet as wn
import re

# Ensure required resources
try:
    _ = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

def tokenize_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def split_context(text: str, level: str):
    """
    Split context into units based on explanation granularity.

    level ∈ {"word", "sentence", "paragraph"}
    """
    text = (text or "").strip()
    if not text:
        return []

    level = level.lower()

    if level == "sentence":
        return tokenize_sentences(text)

    if level == "word":
        return [w for w in text.split() if w.strip()]

    if level == "paragraph":
        return [p.strip() for p in text.split("\n") if p.strip()]

    # if level == "phrase":
    #     # noun phrase chunks (linguistically meaningful)
    #     doc = nlp(text)
    #     return [chunk.text.strip() for chunk in doc.noun_chunks]

    raise ValueError(f"Unknown explanation_level: {level}")


def leave_one_out(tokens: List[str]) -> List[str]:
    perturbs = []
    for i in range(len(tokens)):
        new = tokens[:i] + tokens[i+1:]
        perturbs.append(" ".join(new))
    return perturbs

def random_noise(tokens: List[str], vocabulary: List[str], repeats=3) -> List[str]:
    perturbs = []
    for i in range(len(tokens)):
        for _ in range(repeats):
            new = tokens.copy()
            insert_idx = random.randint(0, len(tokens))
            new.insert(insert_idx, random.choice(vocabulary))
            perturbs.append(" ".join(new))
    return perturbs

def entity_manipulation(sent: str, random_vocab: List[str], repeats=2) -> List[str]:
    doc = nlp(sent)
    ents = [ent for ent in doc.ents]
    perturbs = []
    if not ents:
        # fallback to replace nouns
        nouns = [tok for tok in doc if tok.pos_ in ("NOUN", "PROPN")]
        ents = nouns
    for ent in ents:
        for _ in range(repeats):
            repl = random.choice(random_vocab)
            perturbed = sent.replace(ent.text, repl)
            perturbs.append(perturbed)
    return perturbs

def _wn_replacements(word: str, mode: str) -> List[str]:
    syns = set()
    for s in wn.synsets(word):
        for l in s.lemmas():
            if mode == "synonym":
                syns.add(l.name().replace("_", " "))
            elif mode == "antonym":
                if l.antonyms():
                    syns.update(a.name().replace("_", " ") for a in l.antonyms())
    return list(syns)

def antonym_injection(sent: str) -> List[str]:
    tokens = [tok.text for tok in nlp(sent)]
    perturbs = []
    for i, t in enumerate(tokens):
        ants = _wn_replacements(t, "antonym")
        if ants:
            for a in ants[:2]:
                new = tokens.copy()
                new[i] = a
                perturbs.append(" ".join(new))
    return perturbs

def synonym_injection(sent: str) -> List[str]:
    tokens = [tok.text for tok in nlp(sent)]
    perturbs = []
    for i, t in enumerate(tokens):
        syns = _wn_replacements(t, "synonym")
        if syns:
            for s in syns[:2]:
                new = tokens.copy()
                new[i] = s
                perturbs.append(" ".join(new))
    return perturbs

def reorder_manipulation(sent: str) -> List[str]:
    tokens = sent.split()
    if len(tokens) < 2:
        return []
    perm = tokens.copy()
    random.shuffle(perm)
    return [" ".join(perm)]

def perturb_sentence(sent: str, strategy: str, vocab: List[str]=None) -> List[str]:
    if strategy == "leave_one_out":
        return leave_one_out(sent.split())
    if strategy == "random_noise":
        return random_noise(sent.split(), vocab or ["SOME","RND","WORD"])
    if strategy == "entity_perturber":
        return entity_manipulation(sent, vocab or ["Foo", "Bar", "Baz"])
    if strategy == "antonym_perturber":
        return antonym_injection(sent)
    if strategy == "synonym_perturber":
        return synonym_injection(sent)
    if strategy == "reorder_perturber":
        return reorder_manipulation(sent)
    raise ValueError(f"unknown strategy {strategy}")
