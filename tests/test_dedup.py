import pytest
from src.dedup import deduplicate_chunks, compute_ngrams, compute_overlap

# ============================
# UTIL FUNCTIONS TEST
# ============================

def test_compute_ngrams_basic():
    text = "this is a simple text"
    trigrams = compute_ngrams(text, 3)
    expected = {
        "this is a",
        "is a simple",
        "a simple text"
    }
    assert trigrams == expected

def test_compute_ngrams_short_text():
    # text shorter than n -> empty set
    assert compute_ngrams("two words", 3) == set()

def test_compute_overlap_basic():
    ngrams = {"a b c", "b c d"}
    seen = {"b c d", "c d e"}
    overlap = compute_overlap(ngrams, seen)
    assert overlap == 1/2  # just "b c d" is in common

def test_compute_overlap_empty():
    assert compute_overlap(set(), {"a b c"}) == 0.0
    assert compute_overlap({"a b c"}, set()) == 0.0

# ============================
# PUBLIC FUNCTION TEST
# ============================

def test_extension_replaces_shorter_chunk():
    chunks = [
        "this is a test",
        "this is a test made with pytest",
        "another different chunk",
    ]

    result = deduplicate_chunks(chunks, trigram_size=3, overlap_threshold=0.3)

    assert result == "this is a test made with pytest another different chunk"



def test_shorter_similar_chunk_is_discarded():
    chunks = [
        "this is a test made with pytest",
        "this is a test",
    ]

    result = deduplicate_chunks(chunks, trigram_size=3, overlap_threshold=0.3)

    assert result == "this is a test made with pytest"



def test_distinct_chunks_are_preserved():
    chunks = [
        "this is a test",
        "completely unrelated sentence",
    ]

    result = deduplicate_chunks(chunks, trigram_size=3, overlap_threshold=0.3)

    assert result == "this is a test completely unrelated sentence"



def test_deduplicate_chunks_no_overlap():
    chunks = ["a b c", "d e f", "g h i"]
    result = deduplicate_chunks(chunks)
    # All chunks are different → all kept
    assert result == "a b c d e f g h i"

def test_deduplicate_chunks_empty_chunk():
    chunks = ["", "a b c"]
    result = deduplicate_chunks(chunks)
    # Empty chunks do not fail
    assert result == " a b c"

def test_deduplicate_chunks_threshold_edge():
    # borderline case for threshold
    chunks = ["a b c d", "b c d e"]  # overlap=0.5
    result = deduplicate_chunks(chunks, trigram_size=3, overlap_threshold=0.5)
    # overlap == threshold → not added
    assert result == "a b c d"
