import logging

# ============================
# TRIGRAM-BASED DEDUPLICATION
# ============================

TRIGRAM_SIZE = 3
DEFAULT_OVERLAP_THRESHOLD = 0.30

WINDOW = 1

logger = logging.getLogger(__name__)


def deduplicate_chunks(chunks, trigram_size: int = TRIGRAM_SIZE,
                       overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD) -> str:
    """
    Removes highly repetitive chunks based on trigram overlap.
    Preserves chunk order, preferring more informative chunks.
    """
    logger.debug(f"Starting deduplication on {len(chunks)} segments.")

    accepted = [] # list of dict

    for chunk in chunks:
        trigrams = compute_ngrams(chunk, trigram_size)

        if not trigrams:
            accepted.append({"text": chunk, "ngrams": trigrams})
            continue

        replaced = False

        # iterate over last WINDOW accepted chunks WITH REAL INDICES
        for idx in range(max(0, len(accepted) - WINDOW), len(accepted)):
            prev = accepted[idx]
            overlap = compute_overlap(trigrams, prev["ngrams"])

            if overlap >= overlap_threshold:
                # choose the most informative chunk
                if len(trigrams) > len(prev["ngrams"]):
                    accepted[idx] = {"text": chunk, "ngrams": trigrams}
                replaced = True
                break

        if not replaced:
            accepted.append({"text": chunk, "ngrams": trigrams})

    return " ".join(item["text"] for item in accepted)


# ============================
# UTILITY FUNCTIONS
# ============================

def compute_ngrams(text: str, n: int) -> set:
    """Returns the set of n-grams (e.g. trigrams) extracted from a text."""
    tokens = text.split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def compute_overlap(current: set, seen: set) -> float:
    """Returns a normalized overlap score between current and previously seen n-grams."""
    if not current or not seen:
        return 0.0
    return len(current & seen) / min(len(current), len(seen))

