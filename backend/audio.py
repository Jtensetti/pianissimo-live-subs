"""Bounded utterance buffering. The speech gate measures energy, not speech probability."""

from collections import deque
from dataclasses import dataclass, field
import asyncio
import time
import uuid

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms
MAX_PACKET_SAMPLES = SAMPLE_RATE
OVERLAP_FRAMES = 60  # 1.2 seconds at 20 ms per frame.


class AudioProtocolError(ValueError):
    pass


class Overloaded(RuntimeError):
    pass


def decode_audio(payload: bytes) -> np.ndarray:
    if not payload or len(payload) % 4 or len(payload) > MAX_PACKET_SAMPLES * 4:
        raise AudioProtocolError("Ljudpaket måste vara 1–16000 float32-sampel (mono, 16000 Hz).")
    audio = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(audio).all() or np.any(np.abs(audio) > 1.01):
        raise AudioProtocolError("Ljudet måste innehålla ändliga float32-värden mellan -1 och 1.")
    return audio


@dataclass
class Job:
    id: str
    audio: np.ndarray
    start: float
    end: float
    final: bool
    generation: int
    continuation: bool = False
    created_at: float = field(default_factory=time.monotonic)


class Segmenter:
    """Partial jobs contain the entire utterance, replacing preceding snapshots.

    A 200 ms pre-roll catches quiet consonants before the energy gate opens.
    Finalize after 600 ms silence or at the absolute 8 second limit. Speech
    already in progress uses a lower energy threshold, retaining weaker endings.
    Forced cuts carry 1.2 seconds of context through pauses shorter than 600 ms;
    the inference worker merges text at that boundary. New subthreshold audio
    after a cut is retained for one final, so even low noise can trigger that
    bounded extra inference. An exact stop or digitally silent tail cannot.
    """

    def __init__(self, sensitivity: int = 3, instant: bool = True):
        if sensitivity not in range(1, 6):
            raise AudioProtocolError("Känslighet måste vara ett heltal mellan 1 och 5.")
        self.threshold = {1: .018, 2: .012, 3: .008, 4: .005, 5: .003}[sensitivity]
        self.continue_threshold = self.threshold / 2
        self.instant = instant
        self.prefix = uuid.uuid4().hex[:12]
        self.counter = 0
        self.generation = 0
        self.position = 0
        self.pending = np.empty(0, dtype=np.float32)
        self.pre_roll = deque(maxlen=10)
        self._clear_carry()
        self._reset_utterance()

    def _clear_carry(self):
        self.carry = []
        self.carry_silence = 0
        self.carry_has_new_signal = False

    def _flush_carry(self):
        job = None
        if self.carry and self.carry_has_new_signal:
            # These new samples follow accepted speech. They must not disappear
            # merely because the energy gate never reopened before stop/pause.
            audio = np.concatenate(self.carry).astype(np.float32, copy=False)
            self.counter += 1
            job = Job(f"{self.prefix}-{self.counter}", audio,
                      (self.position - len(audio)) / SAMPLE_RATE,
                      self.position / SAMPLE_RATE, True, self.generation, True)
        self._clear_carry()
        return job

    def _reset_utterance(self):
        self.parts = []
        self.samples = 0
        self.silence = 0
        self.voiced = 0
        self.last_partial = 0
        self.id = None
        self.continuation = False

    def clear(self):
        self.generation += 1
        self.position += len(self.pending)
        self.pending = np.empty(0, dtype=np.float32)
        self.pre_roll.clear()
        self._clear_carry()
        self._reset_utterance()

    def _job(self, final: bool):
        # A short syllable after a forced cut belongs to already accepted speech;
        # applying the initial noise gate again would silently drop that tail.
        if not self.parts or (not self.continuation and self.voiced < int(.12 * SAMPLE_RATE)):
            return None
        # Keep the received tail, including low-energy consonants and the pause.
        # Energy alone cannot establish where the final spoken word ends.
        audio = np.concatenate(self.parts).astype(np.float32, copy=False)
        return Job(self.id, audio, self.start / SAMPLE_RATE,
                   (self.start + self.samples) / SAMPLE_RATE, final, self.generation, self.continuation)

    def _frame(self, frame: np.ndarray):
        jobs = []
        size = len(frame)
        threshold = self.continue_threshold if self.id is not None or self.carry else self.threshold
        voiced = float(np.sqrt(np.mean(frame * frame))) >= threshold
        if self.id is None:
            if not voiced:
                self.pre_roll.append(frame.copy())
                self.position += size
                # The same pause ends a phrase on either side of a forced cut.
                # This holds at most 1.2s context + less than 600ms quiet audio.
                if self.carry:
                    self.carry.append(frame.copy())
                    self.carry_silence += size
                    self.carry_has_new_signal |= bool(np.any(frame))
                    if self.carry_silence >= int(.6 * SAMPLE_RATE):
                        job = self._flush_carry()
                        if job is not None:
                            jobs.append(job)
                return jobs
            self.counter += 1
            self.id = f"{self.prefix}-{self.counter}"
            self.continuation = bool(self.carry)
            self.parts = self.carry or list(self.pre_roll)
            self._clear_carry()
            self.pre_roll.clear()
            self.samples = sum(len(part) for part in self.parts)
            self.start = self.position - self.samples
            if self.continuation:
                # The context alone is not a new partial: wait for fresh speech.
                self.last_partial = self.samples
        self.parts.append(frame.copy())
        self.samples += size
        self.position += size
        if voiced:
            self.voiced += size
            self.silence = 0
        else:
            self.silence += size
        finish = (self.silence >= int(.6 * SAMPLE_RATE)
                  or self.samples >= 8 * SAMPLE_RATE)
        if finish:
            job = self._job(True)
            if job is not None:
                jobs.append(job)
            trailing = min(self.silence, int(.2 * SAMPLE_RATE))
            tail = np.concatenate(self.parts)[-trailing:] if trailing else np.empty(0, dtype=np.float32)
            forced_cut = self.samples >= 8 * SAMPLE_RATE and self.silence < int(.6 * SAMPLE_RATE)
            self._clear_carry()
            self.carry = [part.copy() for part in self.parts[-OVERLAP_FRAMES:]] if forced_cut and job is not None else []
            self.carry_silence = self.silence
            self._reset_utterance()
            for offset in range(0, len(tail), FRAME_SAMPLES):
                self.pre_roll.append(tail[offset:offset + FRAME_SAMPLES])
        elif (self.instant and self.samples >= int(1.2 * SAMPLE_RATE)
              and self.samples - self.last_partial >= int(.6 * SAMPLE_RATE)):
            job = self._job(False)
            if job is not None:
                jobs.append(job)
                self.last_partial = self.samples
        return jobs

    def feed(self, audio: np.ndarray):
        audio = np.concatenate((self.pending, audio))
        stop = len(audio) - len(audio) % FRAME_SAMPLES
        jobs = []
        for offset in range(0, stop, FRAME_SAMPLES):
            jobs.extend(self._frame(audio[offset:offset + FRAME_SAMPLES]))
        self.pending = audio[stop:].copy()
        return jobs

    def flush(self):
        jobs = []
        if len(self.pending):
            jobs.extend(self._frame(self.pending))
            self.pending = np.empty(0, dtype=np.float32)
        job = self._job(True)
        if job is not None:
            jobs.append(job)
        carry_job = self._flush_carry()
        if carry_job is not None:
            jobs.append(carry_job)
        self._reset_utterance()
        self.pre_roll.clear()
        return jobs


class JobBuffer:
    """Coalesce partials; never silently discard finals; at most 4 jobs pending."""

    def __init__(self, max_jobs: int = 4):
        self.jobs = deque()
        self.max_jobs = max_jobs
        self.wake = asyncio.Event()
        self.closed = False

    def submit(self, job: Job):
        self.jobs = deque(old for old in self.jobs if old.final or old.id != job.id)
        if len(self.jobs) >= self.max_jobs:
            if not job.final:
                return  # Final will include all of the utterance audio.
            self.jobs = deque(old for old in self.jobs if old.final)
            if len(self.jobs) >= self.max_jobs:
                raise Overloaded("Datorn hinner inte transkribera ljudet. Inspelningen har stoppats; stäng andra tunga program och försök igen.")
        self.jobs.append(job)
        self.wake.set()

    def clear(self):
        self.jobs.clear()

    def close(self, discard: bool = False):
        self.closed = True
        if discard:
            self.clear()
        self.wake.set()

    async def get(self):
        while not self.jobs:
            if self.closed:
                return None
            self.wake.clear()
            await self.wake.wait()
        return self.jobs.popleft()
