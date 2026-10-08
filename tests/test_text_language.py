from turbo_lyrics.language import detect_text_language
from turbo_lyrics.text import CharMapper, ends_sentence, fold, is_hallucination


def test_fold_and_hallucinations():
    assert fold("Hello, World!  It's ME") == "hello world it's me"
    assert is_hallucination("Untertitel der Amara.org-Community")
    assert is_hallucination("  ")
    assert not is_hallucination("Ich liebe dich")


def test_sentence_end():
    assert ends_sentence("dich.") and ends_sentence('"ja!"') and not ends_sentence("und")


def test_charmapper_lowercase_vocab_with_umlauts_and_eszett():
    vocab = {"<pad>": 0, "|": 4, "'": 5, **{c: i + 6 for i, c in enumerate("abcdefghijklmnopqrstuvwxyzäöü")}}
    m = CharMapper(vocab, blank=0)
    assert not m.upper
    assert m.word_ids("Füße,") == [vocab["f"], vocab["ü"], vocab["s"], vocab["s"], vocab["e"]]
    assert m.word_ids("Café") == [vocab[c] for c in "cafe"]
    assert m.word_ids("—") == []
    assert m.word_ids("'") == []


def test_charmapper_uppercase_vocab():
    vocab = {"<pad>": 0, "|": 4, "'": 27, **{c: i + 5 for i, c in enumerate("ETAOINHSRDLUMWCFGYPBVKXJQZ")}}
    m = CharMapper(vocab, blank=0)
    assert m.upper
    assert m.word_ids("don’t") == [vocab[c] for c in "DON'T"]


def test_detect_text_language():
    de = ["Ich hab dich so vermisst", "und du bist nicht bei mir", "was soll ich nur tun", "wenn es dunkel wird"]
    en = ["I know that you love me", "and it's all for you", "so let me go now", "when the night is over"]
    assert detect_text_language(de * 2) == "de"
    assert detect_text_language(en * 2) == "en"
    assert detect_text_language(["la la la"]) is None


def test_precision_resolution():
    import pytest

    from turbo_lyrics.align import resolve_precision

    assert resolve_precision("auto") in ("fp32", "bf16")
    assert resolve_precision("fp32") == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("int4")
