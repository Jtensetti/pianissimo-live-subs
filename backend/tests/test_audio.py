import asyncio

import numpy as np
import pytest

from backend.audio import AudioProtocolError, Job, JobBuffer, Overloaded, Segmenter, decode_audio, trim_overlap


def sound(seconds, amplitude=.1):
    return np.full(round(seconds * 16000), amplitude, dtype=np.float32)


def test_silence_never_produces_inference_jobs():
    segmenter = Segmenter()
    assert segmenter.feed(sound(20, 0)) == []
    assert segmenter.flush() == []


def test_partial_and_final_share_id_and_include_preroll():
    segmenter = Segmenter()
    assert segmenter.feed(sound(.5, 0)) == []
    partials = segmenter.feed(sound(1.1))
    assert len(partials) == 1
    assert not partials[0].final
    assert partials[0].start == pytest.approx(.3)
    final = segmenter.feed(sound(.6, 0))[-1]
    assert final.final and final.id == partials[0].id
    assert final.end == pytest.approx(1.75)
    assert final.audio.dtype == np.float32
    assert len(final.audio) == 23200
    assert segmenter.flush() == []


def test_stop_flushes_subframe_tail_below_partial_minimum():
    segmenter = Segmenter()
    audio = sound(.213)
    assert segmenter.feed(audio) == []
    final = segmenter.flush()[0]
    assert final.final and final.end == pytest.approx(.213)
    np.testing.assert_array_equal(final.audio, audio)
    assert segmenter.flush() == []


def test_sensitivity_changes_energy_gate():
    assert Segmenter(1).feed(sound(1, .006)) == []
    assert len(Segmenter(5).feed(sound(1.2, .006))) == 1
    with pytest.raises(AudioProtocolError):
        Segmenter(6)


def test_continuous_speech_bounded_with_context_and_unique_ordered_ids():
    segmenter = Segmenter(instant=False)
    jobs = segmenter.feed(sound(20)) + segmenter.flush()
    assert len(jobs) == 3
    assert [job.id.rsplit("-", 1)[1] for job in jobs] == ["1", "2", "3"]
    assert all(len(job.audio) <= 8 * 16000 for job in jobs)
    assert [job.continuation for job in jobs] == [False, True, True]
    assert jobs[1].start == pytest.approx(7.4)
    assert jobs[2].start == pytest.approx(14.8)
    assert jobs[-1].end == pytest.approx(20)


def test_forced_cut_has_no_duplicate_tail_at_exact_stop_or_after_pause():
    segmenter = Segmenter(instant=False)
    assert len(segmenter.feed(sound(8))) == 1
    assert segmenter.flush() == []
    segmenter = Segmenter(instant=False)
    segmenter.feed(sound(8))
    segmenter.feed(sound(.3, 0))
    segmenter.feed(sound(.2))
    job = segmenter.flush()[0]
    assert not job.continuation
    assert job.start == pytest.approx(8.1)


def test_forced_cut_context_survives_a_quiet_consonant():
    segmenter = Segmenter(instant=False)
    segmenter.feed(sound(8))
    segmenter.feed(sound(.04, 0))
    segmenter.feed(sound(.3))
    job = segmenter.flush()[0]
    assert job.continuation
    assert job.start == pytest.approx(7.4)
    assert job.end == pytest.approx(8.34)


def test_short_noise_below_minimum_speech_is_discarded():
    segmenter = Segmenter()
    assert segmenter.feed(sound(.04)) == []
    assert segmenter.feed(sound(.6, 0)) == []


def test_clear_invalidates_old_jobs_and_does_not_reuse_ids():
    segmenter = Segmenter()
    old = segmenter.feed(sound(1.2))[0]
    segmenter.clear()
    new = segmenter.feed(sound(1.2))[0]
    assert old.id != new.id
    assert new.generation == old.generation + 1


@pytest.mark.parametrize("payload", [b"", b"123", bytes(64004),
    np.array([np.nan], dtype="<f4").tobytes(), np.array([np.inf], dtype="<f4").tobytes(),
    np.array([1.2], dtype="<f4").tobytes()], ids=["empty", "odd-bytes", "oversized", "nan", "inf", "range"])
def test_malformed_audio_is_rejected(payload):
    with pytest.raises(AudioProtocolError):
        decode_audio(payload)


def test_audio_protocol_accepts_little_endian_float32():
    expected = np.array([-.5, .25, 0], dtype="<f4")
    np.testing.assert_array_equal(decode_audio(expected.tobytes()), expected)


def test_buffer_coalesces_partials_preserves_finals_and_reports_overload():
    async def scenario():
        buffer = JobBuffer(max_jobs=2)
        def job(name, final=False, duration=.8):
            return Job(name, sound(duration), 0, duration, final, 0)
        buffer.submit(job("a"))
        buffer.submit(job("a", duration=1.4))
        assert len(buffer.jobs) == 1
        buffer.submit(job("a", True, 2))
        buffer.submit(job("b", True, 1))
        buffer.submit(job("c"))
        assert len(buffer.jobs) == 2
        with pytest.raises(Overloaded):
            buffer.submit(job("c", True))
        assert (await buffer.get()).id == "a"
        assert (await buffer.get()).id == "b"
        buffer.close()
        assert await buffer.get() is None
    asyncio.run(scenario())


def test_word_overlap_trimming_preserves_case_and_punctuation_of_new_text():
    assert trim_overlap("Vi går till skogen.", "Till skogen, och hem igen.") == "och hem igen."
    assert trim_overlap("Jag heter Anna", "Anna bor i Malmö.") == "bor i Malmö."
    assert trim_overlap("Samma mening", "Samma mening.") == ""
    assert trim_overlap("Det blir", "en annan början.") == "en annan början."
