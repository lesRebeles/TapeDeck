#!/usr/bin/env python3
"""
cassette.py
================

Store a NES ROM file as audio for cassette tape, and read it back later,
reconstructing a byte-identical .nes file that any emulator can load.

Three tape formats are supported (pass --format to `encode`; `decode`
always auto-detects which one a recording uses). Measured on a 40KB ROM:

  classic  - 300 baud, 1 tone channel. ~27 bytes/sec. The traditional
             Kansas City Standard used by 1970s/80s home computers. No
             error correction beyond a whole-file CRC32 check (detects
             corruption, doesn't recover from it). The most forgiving of
             a rough recording -- this is the format to reach for if a
             transfer keeps failing on the others.

  dual     - 300 baud (the same safe, forgiving symbol rate as classic)
             but with 2 simultaneous tone channels summed together per
             symbol, so every symbol carries 2 data bits instead of 1.
             ~43 bytes/sec (~1.6x classic -- not quite 2x, because the
             3-symbol start/stop framing overhead is fixed per byte
             regardless of channel count, and is proportionally larger
             here). No FEC, same detect-don't-correct contract as
             classic. Recommended default: real speedup, same noise and
             tape-speed-wobble tolerance as classic since the underlying
             symbol duration and cycles-per-tone are unchanged.

  fast     - 1200 baud (4x the symbol rate), 1 tone channel, with
             per-byte Hamming(8,4) SECDED forward error correction so
             single-bit errors are corrected automatically. ~63 bytes/sec
             (~2.3x classic), the fastest option, and the most likely to
             recover from a burst dropout thanks to FEC -- but its short,
             1-2-cycle symbols carry noticeably less margin against
             background noise than the 300-baud formats.

All formats share the same self-clocking start/stop-symbol framing
(every byte re-syncs independently) and the same Goertzel-based tone
detection with a small adaptive per-symbol clock tracker, described
below.

ENCODING
--------
Every format is built from "symbols": one symbol = one audio window of
duration 1/baud seconds. A symbol carries `bits_per_symbol` bits, one per
tone channel active during that window (1 channel for classic/fast, 2 for
dual). Each channel independently plays one of two tones for a '0' or '1':

    channel 0:  1200 Hz = '0'   2400 Hz = '1'
    channel 1:  3600 Hz = '0'   4800 Hz = '1'   (dual format only)

All four frequencies are exact integer multiples of the baud rate, so
every symbol window holds a whole number of cycles of every candidate
tone (no phase clicks between symbols) AND the frequencies are exactly
orthogonal over that window length (like DFT/FFT bins) -- so on the dual
format, channel 0's tone and channel 1's tone can be summed into one
audio signal and cleanly separated again on decode without crosstalk.

  byte frame (both non-FEC and FEC variants):
      start-symbol(all channels=0) + N code symbols + 2 stop-symbols(all=1)

  Non-FEC code symbols carry the raw byte, LSB first, split into groups
  of `bits_per_symbol` bits.

  FEC code symbols carry two Hamming(8,4) SECDED codewords (one per
  nibble of the byte: 4 data + 3 parity + 1 overall-parity bits each,
  16 bits total), split the same way. Each codeword independently
  corrects any single-bit error and detects (without guessing) any
  2-bit error.

  stream = [format tag] [leader tone] [sync bytes] [13-byte header] [ROM bytes] [trailer]
  header = b"NEST1" + uint32 length + uint32 crc32   (little endian)

DECODING
--------
Bit/symbol classification computes exact-bin DFT energy (a batch,
vectorized Goertzel) at each channel's two candidate frequencies and
picks whichever is stronger -- far more robust to noise, DC offset, and
short windows than counting zero crossings.

On top of that, decoding uses a small adaptive per-symbol clock tracker
(an "early-late gate", the same idea a hardware FSK demodulator's PLL
uses): for each symbol it checks a handful of nearby window placements
and keeps whichever gives the strongest, least-ambiguous tone match, then
advances from THAT position rather than a fixed nominal offset. This is
what lets a format keep working through the speed wobble of a real tape
transport instead of silently drifting out of alignment within a byte.
The tracker only actually moves off the nominal (on-schedule) position
when some other offset is a clearly, substantially better match --
otherwise ordinary background noise would occasionally look marginally
better a few samples off and pull the clock estimate around for no
reason, which turned out to hurt plain noise robustness more than it
helped.

USAGE
-----
  Write a ROM to audio and play it live through your speakers/line-out
  (plug that output into a tape recorder's MIC/LINE-IN and hit record):

      python nes_cassette.py encode game.nes --play
      python nes_cassette.py encode game.nes --play --format fast

  Or just render it to a WAV file first (recommended -- check it, then
  play the WAV with any player into the tape recorder):

      python nes_cassette.py encode game.nes --out game_tape.wav --verify
      python nes_cassette.py play game_tape.wav

  Read a tape back. Either record live from your line-in/mic while the
  tape plays:

      python nes_cassette.py decode --record 60 --out restored.nes

  ...or decode a WAV you already captured (e.g. with Audacity). Format
  is auto-detected from the recording:

      python nes_cassette.py decode --wav captured.wav --out restored.nes

DEPENDENCIES
------------
  pip install numpy sounddevice

  numpy is required always. sounddevice is only needed for --play and
  --record (live audio I/O); decoding/encoding to WAV files works with
  just numpy + the standard library `wave` module.
"""

import argparse
import struct
import sys
import wave
import zlib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tape format constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000           # Hz -- divides evenly by all baud rates below

MAGIC = b"NEST1"
SYNC_BYTE = 0x16               # classic "SYN" sync character
NUM_SYNC_BYTES = 8
LEADER_SECONDS = 4.0
TRAILER_SECONDS = 1.0


class Profile:
    """Everything that differs between tape formats: symbol rate, tone
    channels, and whether Hamming FEC is used."""

    def __init__(self, name, baud, channel_freqs, use_fec):
        self.name = name
        self.baud = baud
        self.use_fec = use_fec
        self.channel_freqs = channel_freqs          # [(f0, f1), ...]
        self.n_channels = len(channel_freqs)
        self.samples_per_symbol = SAMPLE_RATE // baud
        assert SAMPLE_RATE % baud == 0, f"{SAMPLE_RATE} must divide evenly by {baud} baud"
        for f0, f1 in channel_freqs:
            assert f0 % baud == 0 and f1 % baud == 0, \
                f"channel frequencies {f0}/{f1} must be exact multiples of {baud} baud"
        self.code_bits_per_byte = 16 if use_fec else 8
        assert self.code_bits_per_byte % self.n_channels == 0, \
            "code bits per byte must divide evenly by the number of channels"
        self.code_symbols_per_byte = self.code_bits_per_byte // self.n_channels
        self.frame_symbols = 1 + self.code_symbols_per_byte + 2   # start + code + 2 stop

    def __repr__(self):
        return (f"Profile({self.name}, {self.baud} baud, {self.n_channels} channel(s), "
                f"fec={self.use_fec})")


PROFILES = {
    "classic": Profile("classic", baud=300, channel_freqs=[(1200, 2400)], use_fec=False),
    "fast": Profile("fast", baud=1200, channel_freqs=[(1200, 2400)], use_fec=True),
    "dual": Profile("dual", baud=300, channel_freqs=[(1200, 2400), (3600, 4800)], use_fec=False),
}

FORMAT_TAG_BAUD = 300
FORMAT_TAG_VALUES = {"classic": 0xC1, "fast": 0xFA, "dual": 0xD2}
FORMAT_TAG_LOOKUP = {v: k for k, v in FORMAT_TAG_VALUES.items()}


# ---------------------------------------------------------------------------
# Hamming(8,4) SECDED forward error correction (used by FEC profiles)
# ---------------------------------------------------------------------------
#
# Each nibble (4 data bits d1..d4) is encoded to 8 bits:
#   position (1-indexed): 1   2   3   4   5   6   7   8
#   content:              p1  p2  d1  p4  d2  d3  d4  p0
#
#   p1 = d1 ^ d2 ^ d4      (covers positions 1,3,5,7)
#   p2 = d1 ^ d3 ^ d4      (covers positions 2,3,6,7)
#   p4 = d2 ^ d3 ^ d4      (covers positions 4,5,6,7)
#   p0 = parity of all 7 preceding bits (adds double-error detection)
#
# The 3-bit syndrome (s1,s2,s4) points directly at the failed position
# (1-7) if exactly one of the first 7 bits is wrong. Combined with the
# overall parity bit this corrects any single-bit error and reliably
# flags (without mis-correcting) any double-bit error.

def hamming84_encode(nibble):
    d1 = (nibble >> 0) & 1
    d2 = (nibble >> 1) & 1
    d3 = (nibble >> 2) & 1
    d4 = (nibble >> 3) & 1
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    bits7 = [p1, p2, d1, p4, d2, d3, d4]
    p0 = 0
    for b in bits7:
        p0 ^= b
    return bits7 + [p0]


def hamming84_decode(bits8):
    """Returns (nibble, status) where status is 'ok', 'corrected', or
    'uncorrectable'."""
    bits7 = list(bits8[:7])
    p1, p2, d1, p4, d2, d3, d4 = bits7
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s4 = p4 ^ d2 ^ d3 ^ d4
    syndrome = s1 | (s2 << 1) | (s4 << 2)
    overall_parity = 0
    for b in bits8:
        overall_parity ^= b

    status = "ok"
    if syndrome == 0 and overall_parity == 0:
        pass
    elif syndrome != 0 and overall_parity == 1:
        idx = syndrome - 1
        bits7[idx] ^= 1
        status = "corrected"
    elif syndrome == 0 and overall_parity == 1:
        status = "corrected"
    else:
        return None, "uncorrectable"

    p1, p2, d1, p4, d2, d3, d4 = bits7
    nibble = d1 | (d2 << 1) | (d3 << 2) | (d4 << 3)
    return nibble, status


# ---------------------------------------------------------------------------
# Encoding: bytes -> symbol stream -> audio samples
# ---------------------------------------------------------------------------
#
# A "symbol" is represented as a tuple of ints, one bit per channel, e.g.
# (0,) for classic/fast or (1, 0) for dual.

def _byte_to_code_bits(byte_val, profile):
    if profile.use_fec:
        low_nibble = byte_val & 0x0F
        high_nibble = (byte_val >> 4) & 0x0F
        return hamming84_encode(low_nibble) + hamming84_encode(high_nibble)
    return [(byte_val >> i) & 1 for i in range(8)]


def _bits_to_symbols(bits, n_channels):
    return [tuple(bits[i:i + n_channels]) for i in range(0, len(bits), n_channels)]


def _byte_to_framed_symbols(byte_val, profile):
    start_symbol = (0,) * profile.n_channels
    stop_symbol = (1,) * profile.n_channels
    code_bits = _byte_to_code_bits(byte_val, profile)
    code_symbols = _bits_to_symbols(code_bits, profile.n_channels)
    return [start_symbol] + code_symbols + [stop_symbol, stop_symbol]


def _build_payload(rom_bytes):
    header = MAGIC + struct.pack("<I", len(rom_bytes)) + struct.pack("<I", zlib.crc32(rom_bytes))
    return header + rom_bytes


def _symbols_for_stream(payload_bytes, profile):
    symbols = []
    leader_symbol = (1,) * profile.n_channels
    symbols.extend([leader_symbol] * int(LEADER_SECONDS * profile.baud))
    for _ in range(NUM_SYNC_BYTES):
        symbols.extend(_byte_to_framed_symbols(SYNC_BYTE, profile))
    for b in payload_bytes:
        symbols.extend(_byte_to_framed_symbols(b, profile))
    symbols.extend([leader_symbol] * int(TRAILER_SECONDS * profile.baud))
    return symbols


def _tone_table(profile):
    """tone_table[channel][bit] = waveform for that channel/bit combo."""
    spb = profile.samples_per_symbol
    t = np.arange(spb) / SAMPLE_RATE
    table = []
    for f0, f1 in profile.channel_freqs:
        tone0 = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        tone1 = np.sin(2 * np.pi * f1 * t).astype(np.float32)
        table.append((tone0, tone1))
    return table


def symbols_to_audio(symbols, profile, amplitude=0.8):
    tone_table = _tone_table(profile)
    spb = profile.samples_per_symbol
    per_channel_amplitude = amplitude / profile.n_channels   # keep peaks in range when summed
    out = np.zeros(len(symbols) * spb, dtype=np.float32)
    for i, symbol in enumerate(symbols):
        mixed = np.zeros(spb, dtype=np.float32)
        for ch, bit in enumerate(symbol):
            mixed += tone_table[ch][bit]
        out[i * spb:(i + 1) * spb] = mixed * per_channel_amplitude
    return out


def _format_tag_audio(format_name, amplitude):
    tag_profile = Profile("tag", baud=FORMAT_TAG_BAUD, channel_freqs=[(1200, 2400)], use_fec=False)
    value = FORMAT_TAG_VALUES[format_name]
    symbols = [(1,)] * int(0.5 * FORMAT_TAG_BAUD)
    symbols += _byte_to_framed_symbols(value, tag_profile)
    symbols += _byte_to_framed_symbols(value, tag_profile)  # send it twice
    return symbols_to_audio(symbols, tag_profile, amplitude=amplitude), tag_profile


def rom_to_audio(rom_bytes, format_name="dual", amplitude=0.8):
    profile = PROFILES[format_name]
    tag_audio, _ = _format_tag_audio(format_name, amplitude)
    payload = _build_payload(rom_bytes)
    symbols = _symbols_for_stream(payload, profile)
    body_audio = symbols_to_audio(symbols, profile, amplitude=amplitude)
    return np.concatenate([tag_audio, body_audio])


# ---------------------------------------------------------------------------
# WAV file helpers (16-bit PCM mono)
# ---------------------------------------------------------------------------

def write_wav(path, samples_float):
    samples_int16 = np.clip(samples_float * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_int16.tobytes())


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    if framerate != SAMPLE_RATE:
        duration = len(data) / framerate
        new_len = int(round(duration * SAMPLE_RATE))
        old_idx = np.linspace(0, len(data) - 1, num=len(data))
        new_idx = np.linspace(0, len(data) - 1, num=new_len)
        data = np.interp(new_idx, old_idx, data)

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Decoding: audio samples -> symbol stream -> bytes
# ---------------------------------------------------------------------------

_GOERTZEL_BASIS_CACHE = {}


def _goertzel_basis(n, freq):
    key = (n, freq)
    basis = _GOERTZEL_BASIS_CACHE.get(key)
    if basis is None:
        k = round(n * freq / SAMPLE_RATE)
        w = 2 * np.pi * k / n
        idx = np.arange(n)
        basis = (np.cos(w * idx).astype(np.float32), np.sin(w * idx).astype(np.float32))
        _GOERTZEL_BASIS_CACHE[key] = basis
    return basis


def _goertzel_power(window, freq):
    """Single-window convenience wrapper (used for leader detection)."""
    n = len(window)
    cos_b, sin_b = _goertzel_basis(n, freq)
    re = np.dot(window, cos_b)
    im = np.dot(window, sin_b)
    return re * re + im * im


def _batch_powers(windows, n, freq):
    """Vectorized energy at `freq` for a whole stack of candidate windows
    at once. windows: shape (M, n)."""
    cos_b, sin_b = _goertzel_basis(n, freq)
    re = windows @ cos_b
    im = windows @ sin_b
    return re * re + im * im


def _classify(window):
    p0 = _goertzel_power(window, PROFILES["classic"].channel_freqs[0][0])
    p1 = _goertzel_power(window, PROFILES["classic"].channel_freqs[0][1])
    return 1 if p1 > p0 else 0


def _classify_for_profile(window, profile):
    """Classify a single window as a symbol tuple under a given profile
    (used for leader detection, which only ever looks at channel 0 -- the
    leader tone is transmitted the same way regardless of channel count)."""
    f0, f1 = profile.channel_freqs[0]
    p0 = _goertzel_power(window, f0)
    p1 = _goertzel_power(window, f1)
    return 1 if p1 > p0 else 0


# How far (in samples) each symbol boundary is allowed to drift from
# where a perfectly steady clock would put it. This is what lets the
# decoder track tape wow/flutter and small motor speed error symbol-by-
# symbol, instead of assuming every symbol is exactly `samples_per_symbol`
# samples long.
_ADAPTIVE_SLACK_FRACTION = 0.15


def _try_decode_byte(samples, start, profile, stats=None):
    """Decode one framed byte starting at sample index `start`, using a
    vectorized per-symbol early-late clock tracker: for each symbol we
    score a small batch of nearby window placements at once (via a single
    matrix multiply per candidate frequency) and keep whichever gives the
    strongest, least-ambiguous combined tone match across all channels,
    then advance from THAT position. Returns (byte_value, next_start) or
    None if the frame doesn't validate."""
    spb = profile.samples_per_symbol
    max_len = len(samples)
    slack = max(1, int(spb * _ADAPTIVE_SLACK_FRACTION))
    offsets = np.arange(-slack, slack + 1)   # center-out isn't needed once we pick the true argmax

    pos = start
    symbols = []
    for _ in range(profile.frame_symbols):
        lo = pos - slack
        hi = pos + slack + spb
        if lo < 0 or hi > max_len:
            return None
        # All candidate windows for this symbol, stacked: shape (2*slack+1, spb)
        window_stack = np.lib.stride_tricks.sliding_window_view(samples[lo:hi], spb)[::1]
        # sliding_window_view over (hi-lo) samples with window spb gives
        # (hi-lo-spb+1) = (2*slack+1) rows, exactly our candidate offsets.

        total_score = np.zeros(window_stack.shape[0], dtype=np.float64)
        per_channel_p0 = []
        per_channel_p1 = []
        for f0, f1 in profile.channel_freqs:
            p0 = _batch_powers(window_stack, spb, f0)
            p1 = _batch_powers(window_stack, spb, f1)
            total_score += np.maximum(p0, p1)
            per_channel_p0.append(p0)
            per_channel_p1.append(p1)

        nominal_idx = slack   # index of delta=0 in offsets/window_stack
        adjusted_score = total_score - 1e-2 * np.abs(offsets)
        best_idx = int(np.argmax(adjusted_score))
        # Only actually move off the nominal position if some other offset
        # is a CLEARLY better match (not just marginally better, which is
        # usually just noise) -- this is what keeps noise robustness for
        # the common case while still letting real speed drift get tracked.
        if adjusted_score[best_idx] <= adjusted_score[nominal_idx] * 1.03:
            best_idx = nominal_idx
        delta = offsets[best_idx]
        bits = tuple(1 if per_channel_p1[c][best_idx] > per_channel_p0[c][best_idx] else 0
                     for c in range(profile.n_channels))
        symbols.append(bits)
        pos = pos + int(delta) + spb

    start_symbol = (0,) * profile.n_channels
    stop_symbol = (1,) * profile.n_channels
    if symbols[0] != start_symbol:
        return None
    if symbols[-2] != stop_symbol or symbols[-1] != stop_symbol:
        return None

    code_bits = [b for sym in symbols[1:-2] for b in sym]

    if profile.use_fec:
        low_bits, high_bits = code_bits[:8], code_bits[8:]
        low_nibble, low_status = hamming84_decode(low_bits)
        high_nibble, high_status = hamming84_decode(high_bits)
        if low_status == "uncorrectable" or high_status == "uncorrectable":
            return None
        if stats is not None:
            if low_status == "corrected":
                stats["corrected"] += 1
            if high_status == "corrected":
                stats["corrected"] += 1
        value = low_nibble | (high_nibble << 4)
    else:
        value = 0
        for i, b in enumerate(code_bits):
            value |= (b << i)

    return value, pos


def _find_first_byte_start(samples, search_from, search_span, profile, stats=None, resync_slop=None):
    if resync_slop is None:
        resync_slop = max(2, profile.samples_per_symbol // 4)
    max_len = len(samples)

    for delta in range(0, resync_slop + 1):
        candidates = (search_from + delta,) if delta == 0 else (search_from + delta, search_from - delta)
        for start in candidates:
            if start < 0 or start + profile.frame_symbols * profile.samples_per_symbol > max_len:
                continue
            result = _try_decode_byte(samples, start, profile, stats=stats)
            if result is not None:
                return result

    lo = max(0, search_from - resync_slop)
    hi = min(max_len, search_from + resync_slop + search_span)
    step = max(1, profile.samples_per_symbol // 8)
    for start in range(lo, hi, step):
        result = _try_decode_byte(samples, start, profile, stats=stats)
        if result is not None:
            return result
    # Fine-grained fallback if the coarse scan above missed a valid frame.
    for start in range(lo, hi):
        result = _try_decode_byte(samples, start, profile, stats=stats)
        if result is not None:
            return result
    return None


def _find_leader_end(samples, profile):
    step = profile.samples_per_symbol
    n_windows = len(samples) // step
    min_run = int(0.3 * profile.baud)
    run_len = 0
    for i in range(n_windows):
        w = samples[i * step:(i + 1) * step]
        bit = _classify_for_profile(w, profile)
        if bit == 1:
            run_len += 1
        else:
            if run_len >= min_run:
                return i * step
            run_len = 0
    if run_len >= min_run:
        return n_windows * step
    return None


def _detect_format(samples):
    tag_profile = Profile("tag", baud=FORMAT_TAG_BAUD, channel_freqs=[(1200, 2400)], use_fec=False)
    leader_end = _find_leader_end(samples, tag_profile)
    if leader_end is None:
        raise ValueError("Could not find the format tag's settling tone at "
                          "the start of the recording.")
    located = _find_first_byte_start(samples, leader_end, tag_profile.samples_per_symbol * 4,
                                      tag_profile, resync_slop=tag_profile.samples_per_symbol * 2)
    if located is None:
        raise ValueError("Could not read the format tag after its leader.")
    value, pos = located
    if value not in FORMAT_TAG_LOOKUP:
        raise ValueError(f"Unrecognized format tag byte 0x{value:02X}.")
    return FORMAT_TAG_LOOKUP[value], pos


def audio_to_rom(samples, verbose=True):
    format_name, tag_end = _detect_format(samples)
    profile = PROFILES[format_name]
    if verbose:
        print(f"[decode] format tag says '{format_name}' ({profile.baud} baud, "
              f"{profile.n_channels} channel(s), fec={'on' if profile.use_fec else 'off'})")

    stats = {"corrected": 0}

    leader_end = _find_leader_end(samples[tag_end:], profile)
    if leader_end is None:
        raise ValueError("Could not find the main leader tone. Is this the "
                          "right recording / is playback level high enough?")
    leader_end += tag_end
    if verbose:
        print(f"[decode] leader ends at sample {leader_end} "
              f"({leader_end / SAMPLE_RATE:.2f}s)")

    located = _find_first_byte_start(samples, leader_end, profile.samples_per_symbol * 4,
                                      profile, stats=stats, resync_slop=profile.samples_per_symbol * 2)
    if located is None:
        raise ValueError("Found a leader tone but could not lock onto the "
                          "first sync byte after it.")
    value, pos = located
    if verbose:
        print(f"[decode] first sync byte 0x{value:02X} located, header starts near sample {pos}")

    synced = 1
    while True:
        result = _find_first_byte_start(samples, pos, profile.samples_per_symbol * 2, profile, stats=stats)
        if result is None:
            break
        value, next_pos = result
        if value == SYNC_BYTE:
            pos = next_pos
            synced += 1
            continue
        break
    if verbose:
        print(f"[decode] consumed {synced} sync bytes, header should start now")

    header_bytes = bytearray()
    cur = pos
    for _ in range(len(MAGIC) + 4 + 4):
        result = _find_first_byte_start(samples, cur, profile.samples_per_symbol * 2, profile, stats=stats)
        if result is None:
            raise ValueError("Lost sync while decoding the header.")
        value, cur = result
        header_bytes.append(value)

    if bytes(header_bytes[:len(MAGIC)]) != MAGIC:
        raise ValueError(f"Bad magic bytes: got {bytes(header_bytes[:len(MAGIC)])!r}, "
                          f"expected {MAGIC!r}. Recording may be corrupt or misaligned.")
    rom_len = struct.unpack("<I", bytes(header_bytes[5:9]))[0]
    expected_crc = struct.unpack("<I", bytes(header_bytes[9:13]))[0]
    if verbose:
        print(f"[decode] header OK: {rom_len} bytes expected, crc32=0x{expected_crc:08X}")

    rom_bytes = bytearray()
    for i in range(rom_len):
        result = _find_first_byte_start(samples, cur, profile.samples_per_symbol * 2, profile, stats=stats)
        if result is None:
            raise ValueError(f"Lost sync while decoding ROM data at byte {i}/{rom_len} "
                              f"({stats['corrected']} bit-errors already corrected before this point).")
        value, cur = result
        rom_bytes.append(value)
        if verbose and rom_len > 0 and i % 4096 == 0:
            pct = 100.0 * i / rom_len
            print(f"[decode] {pct:5.1f}% ({i}/{rom_len} bytes, "
                  f"{stats['corrected']} bit-errors corrected so far)", end="\r")
    if verbose:
        print(f"[decode] 100.0% ({rom_len}/{rom_len} bytes, "
              f"{stats['corrected']} bit-errors corrected total)")

    actual_crc = zlib.crc32(bytes(rom_bytes))
    if actual_crc != expected_crc:
        raise ValueError(f"CRC32 mismatch! expected 0x{expected_crc:08X}, "
                          f"got 0x{actual_crc:08X}. The tape audio has errors "
                          f"beyond what FEC could correct (bad connection, "
                          f"volume too low/high, dirty tape heads, etc).")
    if verbose:
        if profile.use_fec and stats["corrected"]:
            print(f"[decode] CRC32 OK -- ROM reconstructed successfully "
                  f"({stats['corrected']} bit-errors were corrected by FEC).")
        else:
            print("[decode] CRC32 OK -- ROM reconstructed successfully.")

    return bytes(rom_bytes)


# ---------------------------------------------------------------------------
# Live audio I/O (optional; needs `sounddevice`)
# ---------------------------------------------------------------------------

def _require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as e:
        print("This feature needs the 'sounddevice' package AND the system "
              "PortAudio library it depends on.\n"
              f"  (import failed: {e})\n"
              "Install with:  pip install sounddevice\n"
              "If that's already installed, your OS also needs PortAudio itself "
              "(e.g. 'apt install libportaudio2' on Debian/Ubuntu, or "
              "'brew install portaudio' on macOS).", file=sys.stderr)
        sys.exit(1)


def play_samples(samples):
    sd = _require_sounddevice()
    print(f"Playing {len(samples) / SAMPLE_RATE:.1f}s of audio through the "
          f"default output device. Start your tape recorder now, then press "
          f"Enter.")
    input()
    sd.play(samples, SAMPLE_RATE, blocking=True)
    print("Playback finished.")


def record_samples(seconds):
    sd = _require_sounddevice()
    print(f"Recording {seconds:.1f}s from the default input device. "
          f"Start tape playback now, then press Enter.")
    input()
    n_samples = int(seconds * SAMPLE_RATE)
    print("Recording...")
    audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return audio.flatten()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_encode(args):
    rom_path = Path(args.rom)
    rom_bytes = rom_path.read_bytes()
    print(f"Read {len(rom_bytes)} bytes from {rom_path}")

    profile = PROFILES[args.format]
    samples = rom_to_audio(rom_bytes, format_name=args.format, amplitude=args.amplitude)
    duration = len(samples) / SAMPLE_RATE
    print(f"Encoded with '{args.format}' format to {duration:.1f}s of audio "
          f"({profile.baud} baud, {profile.n_channels} channel(s)"
          f"{', Hamming(8,4) SECDED FEC' if profile.use_fec else ''})")

    if args.verify:
        print("Verifying round-trip decode in memory...")
        try:
            decoded = audio_to_rom(samples, verbose=False)
            if decoded == rom_bytes:
                print("Verify OK: decoded audio matches the original ROM exactly.")
            else:
                print("Verify FAILED: decoded bytes differ from the original!",
                      file=sys.stderr)
        except ValueError as e:
            print(f"Verify FAILED: {e}", file=sys.stderr)

    if args.out:
        write_wav(args.out, samples)
        print(f"Wrote WAV file: {args.out}")

    if args.play:
        play_samples(samples)

    if not args.out and not args.play:
        print("Nothing to do -- pass --out FILE.wav and/or --play.")


def cmd_play(args):
    samples = read_wav(args.wav)
    play_samples(samples)


def cmd_decode(args):
    if args.wav:
        samples = read_wav(args.wav)
        print(f"Loaded {len(samples) / SAMPLE_RATE:.1f}s of audio from {args.wav}")
    elif args.record is not None:
        samples = record_samples(args.record)
    else:
        print("Provide either --wav FILE.wav or --record SECONDS", file=sys.stderr)
        sys.exit(1)

    try:
        rom_bytes = audio_to_rom(samples, verbose=True)
    except ValueError as e:
        print(f"Decode failed: {e}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.write_bytes(rom_bytes)
    print(f"Wrote {len(rom_bytes)} bytes to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Store/retrieve a NES ROM as cassette-tape audio (Kansas City Standard FSK).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enc = sub.add_parser("encode", help="Convert a .nes ROM to tape audio.")
    p_enc.add_argument("rom", help="Path to the input .nes ROM file.")
    p_enc.add_argument("--out", help="Write the encoded audio to this WAV file.")
    p_enc.add_argument("--play", action="store_true",
                        help="Play the encoded audio live through the default output device.")
    p_enc.add_argument("--format", choices=["classic", "fast", "dual"], default="dual",
                        help="Tape format: 'dual' = 300 baud, 2 simultaneous tone channels, "
                             "~1.6x classic's speed with the same noise/wobble tolerance "
                             "(default, recommended); 'fast' = 1200 baud, 1 channel, Hamming FEC, "
                             "~2.3x classic's speed but least tolerant of background noise; "
                             "'classic' = original 300 baud Kansas City Standard, most forgiving.")
    p_enc.add_argument("--amplitude", type=float, default=0.8,
                        help="Output amplitude 0..1 (default 0.8).")
    p_enc.add_argument("--verify", action="store_true",
                        help="Immediately decode the generated audio in memory to confirm round-trip integrity.")
    p_enc.set_defaults(func=cmd_encode)

    p_play = sub.add_parser("play", help="Play an already-encoded WAV file (e.g. to feed a tape recorder).")
    p_play.add_argument("wav", help="Path to the WAV file to play.")
    p_play.set_defaults(func=cmd_play)

    p_dec = sub.add_parser("decode", help="Convert tape audio back into a .nes ROM file. "
                                           "Format (classic/fast/dual) is auto-detected.")
    p_dec.add_argument("--wav", help="Decode from an existing WAV file (e.g. captured with Audacity).")
    p_dec.add_argument("--record", type=float, default=None,
                        help="Instead of --wav, record this many seconds live from the default input device.")
    p_dec.add_argument("--out", required=True, help="Path to write the reconstructed .nes ROM file.")
    p_dec.set_defaults(func=cmd_decode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
