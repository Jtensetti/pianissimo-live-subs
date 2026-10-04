"""Conservative stitching of overlapping ASR windows using token emission times.

These are approximate emission times, not word boundaries. An uncertain seam
may repeat words; it must not delete them just because their text is repeated.
"""

from dataclasses import dataclass
import math
import re
import unicodedata


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float


@dataclass(frozen=True)
class Hypothesis:
    text: str
    words: tuple[TimedWord, ...] = ()


MAX_WORDS = 80
TIME_TOLERANCE = .24  # Three Pianissimo encoder steps; never a cutting boundary.
_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")  # onnx-asr's token decoder


def _text(words):
    return " ".join(word.text for word in words)


def cap(hypothesis: Hypothesis, max_words: int = MAX_WORDS) -> Hypothesis:
    """Bound a continuous caption while retaining its available timing evidence."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    if hypothesis.words:
        words = tuple(hypothesis.words[-max_words:])
        return Hypothesis(_text(words), words)
    return Hypothesis(" ".join(hypothesis.text.split()[-max_words:]))


def from_result(result, offset: float = 0.0) -> Hypothesis:
    """Convert onnx-asr's TimestampedResult (or a plain test string).

    Pianissimo emits subwords with a leading space at word starts. Preserve the
    decoder's punctuation joining and use the first letter's token timestamp.
    Incomplete/mismatched timing metadata falls back to text without anchors.
    """
    text = " ".join((result if isinstance(result, str) else result.text).split())
    fallback = Hypothesis(text)
    tokens = getattr(result, "tokens", None)
    times = getattr(result, "timestamps", None)
    if tokens is None or times is None or len(tokens) != len(times):
        return cap(fallback)
    try:
        absolute = [float(time) + offset for time in times]
    except (TypeError, ValueError):
        return cap(fallback)
    if (not all(math.isfinite(time) for time in absolute)
            or any(right < left for left, right in zip(absolute, absolute[1:]))):
        return cap(fallback)
    characters = [(char, time) for token, time in zip(tokens, absolute)
                  for char in token.replace("\u2581", " ")]
    raw = "".join(char for char, _ in characters)
    removed = {index for match in _SPACE_PATTERN.finditer(raw) if not match.group(1)
               for index in range(match.start(), match.end())}
    characters = [char for index, char in enumerate(characters) if index not in removed]
    decoded = "".join(char for char, _ in characters)
    if " ".join(decoded.split()) != text:
        return cap(fallback)
    words = []
    for match in re.finditer(r"\S+", decoded):
        # A separate punctuation token can occur long after the spoken word.
        first = next((index for index in range(match.start(), match.end())
                      if characters[index][0].isalnum()), match.start())
        words.append(TimedWord(match.group(), characters[first][1]))
    return cap(Hypothesis(text, tuple(words)))


def _normal(word: TimedWord) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", word.text).casefold()
                   if char.isalnum())


def stitch(previous: Hypothesis, current: Hypothesis,
           overlap_start: float, overlap_end: float) -> Hypothesis:
    """Replace the old suffix only after finding an unambiguous timed anchor.

    Prefer two adjacent matching words. A single anchor needs at least four
    letters. Times must be inside the actual shared audio, and each anchor must
    have exactly one match in both directions within the timing tolerance.
    Retain the old prefix before the first safe anchor and the entire new suffix
    from that anchor, allowing the new window to repair the old cut-off ending.
    With no timing evidence, concatenate rather than guessing away real speech.
    """
    if not previous.text:
        return cap(current)
    if not current.text:
        return cap(previous)
    if previous.words and current.words and overlap_end > overlap_start:
        old = [_normal(word) for word in previous.words]
        new = [_normal(word) for word in current.words]
        candidates = [(i, j) for i, left in enumerate(previous.words)
                      if overlap_start <= left.start <= overlap_end
                      for j, right in enumerate(current.words)
                      if overlap_start <= right.start <= overlap_end
                      and old[i] and old[i] == new[j]
                      and abs(left.start - right.start) <= TIME_TOLERANCE + 1e-9]
        unique = {(i, j) for i, j in candidates
                  if sum(a == i for a, _ in candidates) == 1
                  and sum(b == j for _, b in candidates) == 1}
        pairs = [(i, j) for i, j in unique if (i + 1, j + 1) in unique]
        singles = [(i, j) for i, j in unique
                   if sum(char.isalpha() for char in old[i]) >= 4]
        anchors = pairs or singles
        if anchors:
            i, j = min(anchors, key=lambda pair: (pair[1], pair[0]))
            words = previous.words[:i] + current.words[j:]
            return cap(Hypothesis(_text(words), words))
    words = previous.words + current.words if previous.words and current.words else ()
    return cap(Hypothesis(previous.text + " " + current.text, words))
