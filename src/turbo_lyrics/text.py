"""Text handling: normalisation for CTC alignment, word splitting, sentence splitting, hallucination filters."""

from __future__ import annotations

import re
import unicodedata

# Whisper's well-known outputs on music/silence (subtitle credits, outros). Compared after normalisation.
HALLUCINATIONS = (
    "untertitel der amara org community",
    "untertitel im auftrag des zdf",
    "untertitel von stephanie geiges",
    "vielen dank fürs zuschauen",
    "danke fürs zuschauen",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by the amara org community",
    "sous titres réalisés par la communauté d amara org",
    "sottotitoli creati dalla comunità amara org",
    "subtítulos realizados por la comunidad de amara org",
    "ondertiteling door de amara org gemeenschap",
)

_SENTENCE_END = re.compile(r"[.!?…。！？]+[\"'»”)\]]*$")


def fold(text: str) -> str:
    """Casefold and strip punctuation (keeps letters/digits of every script, collapses whitespace)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text))


def is_hallucination(text: str) -> bool:
    folded = fold(text)
    if not folded:
        return True
    return any(phrase in folded for phrase in HALLUCINATIONS)


def split_words(text: str) -> list[str]:
    """Whitespace words as WhisperX reports them (punctuation stays attached to the word)."""
    return [w for w in text.split() if w.strip()]


def ends_sentence(word: str) -> bool:
    return bool(_SENTENCE_END.search(word))


class CharMapper:
    """Maps words to the character vocabulary of a CTC model.

    Characters missing from the vocabulary are folded (``é`` -> ``e``, ``ß`` -> ``ss`` when ``ß`` is unknown) and
    otherwise dropped; a word without any alignable character gets its time interpolated later.
    """

    def __init__(self, vocab: dict[str, int], blank: int, delimiter: str | None = "|"):
        self.vocab = vocab
        self.blank = blank
        self.delimiter_id = vocab.get(delimiter) if delimiter else None
        letters = [k for k in vocab if len(k) == 1 and k.isalpha()]
        self.upper = bool(letters) and sum(k.isupper() for k in letters) > len(letters) / 2

    def _char_ids(self, ch: str) -> list[int]:
        c = ch.upper() if self.upper else ch.lower()
        if c in self.vocab:
            return [self.vocab[c]]
        if ch in ("’", "‘", "`", "´") and "'" in self.vocab:
            return [self.vocab["'"]]
        decomposed = unicodedata.normalize("NFKD", c)
        base = "".join(d for d in decomposed if not unicodedata.combining(d))
        if base and base != c:
            ids = []
            for b in base:
                bb = b.upper() if self.upper else b.lower()
                if bb in self.vocab:
                    ids.append(self.vocab[bb])
            if ids:
                return ids
        if c.lower() == "ß":
            s = "SS" if self.upper else "ss"
            if all(x in self.vocab for x in s):
                return [self.vocab[x] for x in s]
        return []

    def word_ids(self, word: str) -> list[int]:
        ids: list[int] = []
        for ch in word:
            if ch.isalnum() or ch in "'’‘`´":
                ids.extend(self._char_ids(ch))
        # an apostrophe alone carries no sound
        if ids and all(i == self.vocab.get("'") for i in ids):
            return []
        return ids
