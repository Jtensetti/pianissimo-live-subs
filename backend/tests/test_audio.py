import asyncio

import numpy as np
import pytest

from backend.audio import AudioProtocolError, Job, JobBuffer, Overloaded, Segmenter, decode_audio


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
    assert final.end == pytest.approx(2.2)
    assert final.audio.dtype == np.float32
    assert len(final.audio) == 30400
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
    assert jobs[1].start == pytest.approx(6.8)
    assert jobs[2].start == pytest.approx(13.6)
    assert jobs[-1].end == pytest.approx(20)


def test_forced_cut_has_no_duplicate_tail_at_exact_stop_or_after_pause():
    segmenter = Segmenter(instant=False)
    assert len(segmenter.feed(sound(8))) == 1
    assert segmenter.flush() == []
    segmenter = Segmenter(instant=False)
    segmenter.feed(sound(8))
    segmenter.feed(sound(.7, 0))
    segmenter.feed(sound(.2))
    job = segmenter.flush()[0]
    assert not job.continuation
    assert job.start == pytest.approx(8.5)


def test_forced_cut_context_survives_a_quiet_consonant():
    segmenter = Segmenter(instant=False)
    segmenter.feed(sound(8))
    segmenter.feed(sound(.04, 0))
    segmenter.feed(sound(.3))
    job = segmenter.flush()[0]
    assert job.continuation
    assert job.start == pytest.approx(6.8)
    assert job.end == pytest.approx(8.34)


def test_short_noise_below_minimum_speech_is_discarded():
    segmenter = Segmenter()
    assert segmenter.feed(sound(.04)) == []
    assert segmenter.feed(sound(.6, 0)) == []


def test_stop_preserves_weak_end_of_word_after_energy_gate_closes():
    segmenter = Segmenter(instant=False)
    audio = np.concatenate((sound(.3), sound(.4, .004)))
    assert segmenter.feed(audio) == []
    final = segmenter.flush()[0]
    np.testing.assert_array_equal(final.audio, audio)
    assert final.end == pytest.approx(.7)


def test_short_last_syllable_after_forced_cut_is_not_discarded_as_noise():
    segmenter = Segmenter(instant=False)
    audio = sound(8.08)
    jobs = segmenter.feed(audio) + segmenter.flush()
    assert len(jobs) == 2
    assert jobs[1].continuation
    assert jobs[1].start == pytest.approx(6.8)
    assert jobs[1].end == pytest.approx(8.08)
    np.testing.assert_array_equal(jobs[1].audio, audio[round(6.8 * 16000):])


def test_forced_cut_preserves_context_when_the_last_frame_is_unvoiced():
    segmenter = Segmenter(instant=False)
    audio = np.concatenate((sound(7.98), sound(.02, 0), sound(.3)))
    jobs = segmenter.feed(audio) + segmenter.flush()
    assert len(jobs) == 2
    assert jobs[1].continuation
    assert jobs[1].start == pytest.approx(6.8)
    np.testing.assert_array_equal(jobs[1].audio, audio[round(6.8 * 16000):])


def test_brief_pause_after_six_seconds_keeps_one_complete_utterance():
    segmenter = Segmenter(instant=False)
    audio = np.concatenate((sound(6.2), sound(.12, 0), sound(.3)))
    jobs = segmenter.feed(audio) + segmenter.flush()
    assert len(jobs) == 1
    np.testing.assert_array_equal(jobs[0].audio, audio)


def test_active_speech_hysteresis_keeps_a_weak_syllable_without_opening_on_noise():
    segmenter = Segmenter(instant=False)
    assert segmenter.feed(sound(.7, .006)) == []  # Below the opening threshold.
    segmenter.clear()
    audio = np.concatenate((sound(.3), sound(.8, .006), sound(.2)))
    assert segmenter.feed(audio) == []  # Weak middle syllable must not endpoint.
    np.testing.assert_array_equal(segmenter.flush()[0].audio, audio)


def test_forced_cut_carries_context_through_short_word_pause_without_sample_loss():
    segmenter = Segmenter(instant=False)
    audio = np.concatenate((sound(8), sound(.5, 0), sound(.08)))
    jobs = segmenter.feed(audio) + segmenter.flush()
    assert len(jobs) == 2
    assert jobs[1].continuation and jobs[1].start == pytest.approx(6.8)
    assert jobs[1].end == pytest.approx(8.58)
    np.testing.assert_array_equal(jobs[1].audio, audio[round(6.8 * 16000):])


@pytest.mark.parametrize("pause", [.04, .3, .6, 1.0])
def test_exact_forced_cut_then_pause_and_stop_does_not_emit_duplicate_final(pause):
    segmenter = Segmenter(instant=False)
    assert len(segmenter.feed(sound(8))) == 1
    assert segmenter.feed(sound(pause, 0)) == []
    assert segmenter.flush() == []


def test_phrase_pause_counts_silence_on_both_sides_of_forced_cut():
    segmenter = Segmenter(instant=False)
    jobs = segmenter.feed(np.concatenate((sound(7.5), sound(.5, 0))))
    assert len(jobs) == 1
    segmenter.feed(sound(.1, 0))
    segmenter.feed(sound(.2))
    following = segmenter.flush()[0]
    assert not following.continuation  # 600ms pause, including the 500ms before cut.
    assert following.start == pytest.approx(7.9)


def test_forced_cut_waits_for_fresh_audio_before_partial_redecoding():
    segmenter = Segmenter()
    assert segmenter.feed(sound(8))[-1].final
    assert segmenter.feed(sound(.02)) == []
    partial = segmenter.feed(sound(.58))[0]
    assert partial.continuation and not partial.final
    assert partial.end == pytest.approx(8.6)
    assert len(partial.audio) == round(1.8 * 16000)


def test_final_jobs_cover_every_sample_of_continuous_speech_with_uneven_packets():
    segmenter = Segmenter(instant=False)
    audio = np.linspace(.05, .15, round(16.087 * 16000), dtype=np.float32)
    jobs = []
    for offset in range(0, len(audio), 1237):
        jobs.extend(segmenter.feed(audio[offset:offset + 1237]))
    jobs.extend(segmenter.flush())
    covered = np.zeros(len(audio), dtype=bool)
    for job in jobs:
        start = round(job.start * 16000)
        end = round(job.end * 16000)
        assert len(job.audio) <= 8 * 16000
        np.testing.assert_array_equal(job.audio, audio[start:end])
        covered[start:end] = True
    assert covered.all()


def test_clear_invalidates_old_jobs_and_does_not_reuse_ids():
    segmenter = Segmenter()
    old = segmenter.feed(sound(1.2))[0]
    segmenter.clear()
    new = segmenter.feed(sound(1.2))[0]
    assert old.id != new.id
    assert new.generation == old.generation + 1


@pytest.mark.parametrize("tail_samples", [1, 219, 6400])
def test_flush_preserves_subthreshold_tail_after_cap_even_below_one_frame(tail_samples):
    segmenter = Segmenter(instant=False)
    audio = np.concatenate((sound(8), np.full(tail_samples, .002, dtype=np.float32)))
    jobs = segmenter.feed(audio) + segmenter.flush()
    assert len(jobs) == 2
    tail = jobs[1]
    assert tail.final and tail.continuation
    assert tail.id.endswith("-2") and tail.generation == jobs[0].generation
    assert tail.start == pytest.approx(6.8)
    assert tail.end == pytest.approx(len(audio) / 16000)
    np.testing.assert_array_equal(tail.audio, audio[round(6.8 * 16000):])
    assert segmenter.flush() == []


def test_subthreshold_tail_emits_one_final_at_pause_then_releases_context():
    segmenter = Segmenter(instant=False)
    first = segmenter.feed(sound(8))[0]
    assert segmenter.feed(sound(.4, .002)) == []
    pause_jobs = segmenter.feed(sound(.6, 0))
    assert len(pause_jobs) == 1
    tail = pause_jobs[0]
    assert tail.continuation and tail.id.endswith("-2")
    assert tail.generation == first.generation
    assert tail.start == pytest.approx(6.8) and tail.end == pytest.approx(8.6)
    np.testing.assert_array_equal(tail.audio[-9600:], np.concatenate((sound(.4, .002), sound(.2, 0))))
    assert len(tail.audio) <= round(1.8 * 16000)
    assert segmenter.flush() == []
    segmenter.feed(sound(.2))
    following = segmenter.flush()[0]
    assert following.id.endswith("-3") and not following.continuation


def test_clear_discards_pending_tail_and_keeps_absolute_sample_position():
    segmenter = Segmenter(instant=False)
    old_audio = np.concatenate((sound(8), np.full(219, .002, dtype=np.float32)))
    old_final = segmenter.feed(old_audio)[0]
    segmenter.clear()
    assert segmenter.flush() == []
    fresh = sound(.213)
    segmenter.feed(fresh)
    new_final = segmenter.flush()[0]
    assert new_final.id.endswith("-2")
    assert new_final.generation == old_final.generation + 1
    assert not new_final.continuation
    assert new_final.start == pytest.approx(len(old_audio) / 16000)
    assert new_final.end == pytest.approx((len(old_audio) + len(fresh)) / 16000)
    np.testing.assert_array_equal(new_final.audio, fresh)


def test_clear_discards_carry_with_weak_frames_already_buffered():
    segmenter = Segmenter(instant=False)
    segmenter.feed(sound(8))
    segmenter.feed(sound(.2, .002))
    segmenter.clear()
    assert segmenter.flush() == []
    segmenter.feed(sound(.2))
    following = segmenter.flush()[0]
    assert not following.continuation and following.generation == 1
    assert following.start == pytest.approx(8.2)
    np.testing.assert_array_equal(following.audio, sound(.2))


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
