"""Language of a lyric text from stopword counts (good enough to pick Whisper's language and the aligner)."""

from __future__ import annotations

from .text import fold

STOPWORDS = {
    "en": "the and you i to a of in it is my me that on your for be all we with so this but love no oh do know not "
    "what can are when just like don't i'm it's was get up go now one out if will",
    "de": "und ich du die der das nicht ist in mit mir dich ein es zu sie wir auf den mich so dir auch was wie "
    "noch nur eine im dem bin hab sein wenn dass für mein von uns doch",
    "fr": "je de la le et les tu à un pas que des en est me une moi ne dans pour qui mon on toi elle il ce vous "
    "mais nous suis plus ma sur",
    "es": "que de la el y me no te en a mi un tu lo es por se una yo con los para si qué como más las al eres "
    "estoy pero mí",
    "it": "di che e non la il mi ti un è a per sei in tu io una me ma con le si come più lo mio da ho cosa del",
    "pt": "que de e o não a eu me um você é do da te se em meu com no na pra mais minha uma por os tu",
    "nl": "ik de je het een en niet is van dat in zijn me mijn op ze wat er maar als met we te voor jij nog",
    "pl": "i nie się w to na że z jak mnie ja co mi a ty jest do tak już ale tylko o by mam cię",
    "sv": "jag och du det att en är i inte på som med mig för har den vi så kan om var till dig men",
    "tr": "bir ve bu ben sen ne de da çok gibi için var yok mi ama seni beni o daha her",
    "ru": "и я не в ты на что с меня мне а как это все так но у ты ж бы мы же по",
}
_SETS = {lang: set(words.split()) for lang, words in STOPWORDS.items()}


def detect_text_language(text: str | list[str], min_hits: int = 8, min_margin: float = 1.5) -> str | None:
    """ISO 639-1 code of the dominant language, or None when the text is too short or ambiguous."""
    if isinstance(text, list):
        text = "\n".join(text)
    words = fold(text).split()
    if not words:
        return None
    counts = {lang: sum(w in vocab for w in words) for lang, vocab in _SETS.items()}
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    (best, hits), (_, second) = ranked[0], ranked[1]
    if hits < min_hits or hits < second * min_margin:
        return None
    return best
