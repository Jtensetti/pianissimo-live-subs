from types import SimpleNamespace

import pytest

from backend.transcript import Hypothesis, TimedWord, cap, from_result, stitch


def hypothesis(text, times):
    return Hypothesis(text, tuple(TimedWord(word, time) for word, time in zip(text.split(), times)))


def test_real_pianissimo_subwords_become_whole_words_with_absolute_times():
    result = SimpleNamespace(
        text="Diskmaskinen gick sönder.",
        tokens=[" D", "isk", "mas", "k", "inen", " g", "ick", " s", "ö", "nd", "er", " ."],
        timestamps=[1.2, 1.36, 1.68, 1.84, 2, 2.16, 2.24, 2.4, 2.48, 2.48, 2.56, 3.2],
    )
    parsed = from_result(result, 6.8)
    assert parsed.text == result.text
    assert [word.text for word in parsed.words] == ["Diskmaskinen", "gick", "sönder."]
    assert [word.start for word in parsed.words] == pytest.approx([8, 8.96, 9.2])


@pytest.mark.parametrize("times", [None, [.1], [.1, float("nan")], [.2, .1]])
def test_missing_or_invalid_timestamps_keep_text_without_alignment(times):
    result = SimpleNamespace(text="Hej världen", tokens=[" Hej", " världen"], timestamps=times)
    assert from_result(result) == Hypothesis("Hej världen")


def test_mismatched_text_and_tokens_do_not_invent_alignment():
    result = SimpleNamespace(text="Hej världen", tokens=[" Hallå", " världen"], timestamps=[.1, .3])
    assert from_result(result) == Hypothesis("Hej världen")


def test_pair_anchor_repairs_old_truncated_ending_and_spelling():
    previous = hypothesis("Vi pratade om den nya diskmask", [4, 6, 6.5, 6.9, 7.2, 7.7])
    current = hypothesis("om den nya diskmaskinen idag.", [6.58, 6.98, 7.28, 7.78, 8.3])
    merged = stitch(previous, current, 6.8, 8)
    assert merged.text == "Vi pratade om den nya diskmaskinen idag."
    assert merged.words[-2].start == 7.78


def test_a_later_pair_is_preferred_over_an_isolated_long_word():
    previous = hypothesis("först katten och den nya saken", [5, 6.85, 7.05, 7.2, 7.5, 7.8])
    current = hypothesis("katten vid den nya saken här", [6.93, 7.1, 7.28, 7.58, 7.88, 8.3])
    assert stitch(previous, current, 6.8, 8).text == "först katten och den nya saken här"


def test_distinct_repeated_word_after_overlap_is_not_deleted():
    previous = hypothesis("Det var väldigt", [6, 6.5, 7.92])
    current = hypothesis("väldigt bra", [8.08, 8.5])
    assert stitch(previous, current, 6.8, 8).text == "Det var väldigt väldigt bra"


def test_short_single_word_is_never_an_anchor():
    previous = hypothesis("ja", [7.5])
    current = hypothesis("ja jag kommer", [7.58, 8.1, 8.5])
    assert stitch(previous, current, 6.8, 8).text == "ja ja jag kommer"


def test_ambiguous_repeated_long_word_keeps_both_hypotheses():
    previous = hypothesis("mycket mycket", [7.3, 7.5])
    current = hypothesis("mycket bättre", [7.4, 8.1])
    assert stitch(previous, current, 6.8, 8).text == "mycket mycket mycket bättre"


def test_two_adjacent_short_words_can_anchor_with_unique_times():
    previous = hypothesis("Det är ju en cyk", [5, 6.3, 6.9, 7.3, 7.7])
    current = hypothesis("ju en cykel", [6.98, 7.38, 7.78])
    assert stitch(previous, current, 6.8, 8).text == "Det är ju en cykel"


def test_single_unique_long_word_anchors_with_case_and_punctuation_changes():
    previous = hypothesis("Jag åker Stockholm.", [5, 6, 7.3])
    current = hypothesis("stockholm imorgon", [7.38, 8.1])
    assert stitch(previous, current, 6.8, 8).text == "Jag åker stockholm imorgon"


def test_text_match_outside_shared_audio_or_timing_tolerance_is_retained():
    previous = hypothesis("kanske", [7.3])
    current = hypothesis("kanske", [7.7])
    assert stitch(previous, current, 6.8, 8).text == "kanske kanske"
    assert stitch(previous, previous, 7.5, 8).text == "kanske kanske"


def test_plain_string_fallback_concatenates_without_text_only_deduplication():
    assert stitch(from_result("ja"), from_result("ja"), 0, 1) == Hypothesis("ja ja")


def test_empty_updates_preserve_previous_text():
    previous = hypothesis("färdig", [1])
    assert stitch(previous, Hypothesis(""), 0, 2) == previous
    assert stitch(Hypothesis(""), previous, 0, 2) == previous


def test_continuous_caption_is_bounded_with_its_timestamps():
    previous = hypothesis(" ".join(f"ord{i}" for i in range(80)), list(range(80)))
    current = hypothesis("nya ord", [81, 82])
    merged = stitch(previous, current, 79, 80)
    assert len(merged.words) == 80
    assert merged.words[0] == TimedWord("ord2", 2)
    assert merged.text.endswith("nya ord")
    assert len(cap(from_result(" ".join("ord" for _ in range(90)))).text.split()) == 80
