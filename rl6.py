#!/usr/bin/env python3
"""
retro_launchpad.py
====================

A blue-screen, C64-style terminal "launchpad" for the cassette tape
storage format implemented in nes_cassette.py: listen to a tape (real
hardware, or a WAV file for testing), reassemble an Atari 2600 ROM, and
launch it straight into RetroArch (Stella core).

INSTALLATION
------------
Drop this file AND nes_cassette.py into your RetroArch install directory
(the one containing your "cores" folder). The launchpad always runs
from its own location -- it changes its working directory to wherever
this script lives on startup -- so it finds RetroArch and the cores
folder correctly no matter where you launch it from.

    <your retroarch dir>/
        retroarch(.exe)
        cores/
            stella_libretro.so   (or .dll / .dylib)
        retro_launchpad.py       <- this file
        nes_cassette.py
        settings.conf            <- created automatically on first run

SETTINGS
--------
All configuration is persisted to settings.conf (plain INI, sits next to
this script) so it survives across runs. On first run, RetroArch's path
and the cores folder are auto-detected by checking real, common locations
for your OS -- a native package install (e.g. `retroarch` on PATH, cores
in /usr/lib64/libretro on Fedora/Nobara), Flatpak, macOS's Application
Support folder, Windows' Program Files/AppData, or a portable install
kept right next to this script -- and picking whichever one actually
exists. Nothing here is hardcoded to one OS or install method; change
anything in the SETTINGS menu (or edit settings.conf directly) if the
auto-detected value is wrong for your setup.

MENU OPTIONS
------------
  1) LISTEN TO TAPE & LOAD    - record live audio (mic/line-in, or a
                                 loopback/"stereo mix" device for a fully
                                 self-contained test with no physical
                                 tape at all) and decode it into a ROM.
  2) LOAD FROM WAV FILE        - decode an already-captured WAV.
  3) SAVE ROM TO TAPE          - encode a ROM to audio and ALWAYS write a
                                 real .wav file to disk first (so you can
                                 play it back through system audio later
                                 for testing), then optionally play it.
  4) PLAY WAV THROUGH SPEAKERS - play any WAV out loud (e.g. to feed a
                                 real tape recorder, or to test LISTEN).
  5) LAUNCH IN RETROARCH       - launch the last-loaded (or any) ROM in
                                 RetroArch with the Stella (Atari 2600) core.
  6) SETTINGS                  - configure RetroArch/cores paths, persisted
                                 to settings.conf.
  7) QUIT

SELF-CONTAINED TEST LOOP (no physical tape/cable needed)
---------------------------------------------------------
Pick your OS's audio loopback device (Windows "Stereo Mix", Linux
PulseAudio/PipeWire "Monitor of ...", macOS needs a virtual device like
BlackHole) as the input device in option 1. Then either:

  a) run option 3 first to create a tape WAV, start option 1 (LISTEN),
     and from a *second* terminal run option 4 to play that WAV -- the
     loopback device will feed it straight back into LISTEN; or
  b) inside option 1 itself, answer the "start test playback" prompt
     with the path to a tape WAV -- this plays it in the background on
     the default output device while recording continues on the chosen
     input device, all in one run.

DEPENDENCIES
------------
  pip install numpy sounddevice
"""

import configparser
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import nes_cassette as nc

# ---------------------------------------------------------------------------
# Retro C64-style terminal theme
# ---------------------------------------------------------------------------

BG_BLUE = "\x1b[44m"
FG_WHITE = "\x1b[1;97m"
RESET = "\x1b[0m"
CLEAR = "\x1b[2J\x1b[H"

WIDTH = 64


def theme_on():
    sys.stdout.write(BG_BLUE + FG_WHITE)
    sys.stdout.flush()


def theme_off():
    sys.stdout.write(RESET + "\n")
    sys.stdout.flush()


def cls():
    sys.stdout.write(CLEAR + BG_BLUE + FG_WHITE)
    sys.stdout.flush()


def typewriter(text, delay=0.008):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if ch != "\n":
            time.sleep(delay)
    sys.stdout.write("\n")


def hr(ch="*"):
    print(ch * WIDTH)


def center(text):
    print(text.center(WIDTH))


def boot_screen():
    cls()
    typewriter("")
    center("**** ATARI 2600 CASSETTE LAUNCHPAD BASIC V2 ****")
    center("64K RAM SYSTEM  38911 TAPE BYTES FREE")
    print()
    typewriter("LOADING KERNEL" + "." * 3, delay=0.05)
    typewriter("SEARCHING FOR CASSETTE INTERFACE" + "." * 3, delay=0.03)
    typewriter("FOUND.", delay=0.03)
    print()
    time.sleep(0.2)


def draw_menu(cfg, last_loaded):
    cls()
    hr("=")
    center("ATARI 2600 CASSETTE LAUNCHPAD v1.0")
    hr("=")
    print()
    print("  1) LISTEN TO TAPE & LOAD")
    print("  2) LOAD FROM WAV FILE")
    print("  3) SAVE ROM TO TAPE (CREATE WAV)")
    print("  4) PLAY WAV THROUGH SPEAKERS (TEST)")
    print("  5) LAUNCH IN RETROARCH")
    print("  6) SETTINGS")
    print("  7) QUIT")
    print()
    hr("-")
    loaded_str = str(last_loaded["path"]) if last_loaded["path"] else "(NONE)"
    print(f"LAST LOADED : {loaded_str}")
    print(f"TAPE FORMAT : {cfg['default_format'].upper()}")
    hr("-")
    print()


def pause():
    input("PRESS ENTER TO CONTINUE" + BLINK_CURSOR())


def BLINK_CURSOR():
    return " \u2588"


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}> ").strip()
    return val if val else default


CLEAR_SENTINEL = "<CLEAR>"


def ask_clearable(prompt, current):
    """Like ask(), but for optional fields where the person needs a way to
    explicitly empty them. Blank input keeps the current value (same as
    every other setting); typing the literal <CLEAR> empties it."""
    val = ask(f"{prompt} (type {CLEAR_SENTINEL} to clear)", current)
    if val is not None and val.strip().upper() == CLEAR_SENTINEL:
        return ""
    return val


# ---------------------------------------------------------------------------
# Config persistence -- plain INI file, settings.conf, next to this script.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "settings.conf"


def _default_retroarch_candidates():
    """Ordered by likelihood, most OS-idiomatic first. A bare command name
    (no path) lets shutil.which() resolve it via PATH later, which is
    exactly how a native package-manager install (Nobara/Fedora dnf,
    Debian/Ubuntu apt, etc.) expects to be run: just `retroarch`."""
    exe_name = "retroarch.exe" if sys.platform.startswith("win") else "retroarch"
    candidates = [exe_name]   # PATH lookup first -- covers most native installs
    if sys.platform == "darwin":
        candidates.append("/Applications/RetroArch.app/Contents/MacOS/RetroArch")
    elif sys.platform.startswith("win"):
        candidates.append(str(Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
                               / "RetroArch" / "retroarch.exe"))
    # Portable install (Windows zip, or a manually-built Linux/Mac build kept
    # next to this script) -- always available as a last-resort default.
    candidates.append(str(BASE_DIR / exe_name))
    return candidates


def _default_cores_dir_candidates():
    """Where RetroArch cores actually live varies a lot by OS and install
    method. This checks the common real locations, in priority order, and
    the caller picks the first one that exists on disk."""
    home = Path.home()
    candidates = []
    if sys.platform.startswith("linux"):
        candidates += [
            home / ".config" / "retroarch" / "cores",                       # RetroArch's own default (most installs, incl. Nobara/Fedora)
            home / ".var" / "app" / "org.libretro.RetroArch" / "config" / "retroarch" / "cores",  # Flatpak
            Path("/usr/lib64/libretro"),                                     # Fedora/Nobara/RHEL-family system package
            Path("/usr/lib/libretro"),                                       # other RPM/generic distros
            Path("/usr/lib/x86_64-linux-gnu/libretro"),                      # Debian/Ubuntu multiarch
        ]
    elif sys.platform == "darwin":
        candidates.append(home / "Library" / "Application Support" / "RetroArch" / "cores")
    elif sys.platform.startswith("win"):
        candidates.append(Path(os.environ.get("APPDATA", "")) / "RetroArch" / "cores")
    # Portable install fallback -- always last.
    candidates.append(BASE_DIR / "cores")
    return candidates


def _pick_first_existing(candidates):
    for c in candidates:
        c_str = str(c)
        # A bare command name (no path separator) should be checked against
        # PATH, not treated as a relative file path -- Path("retroarch").exists()
        # would (almost) never be true even when `retroarch` is perfectly
        # runnable from any directory.
        if os.sep not in c_str and (not sys.platform.startswith("win") or "/" not in c_str):
            if shutil.which(c_str):
                return c_str
            continue
        if Path(c_str).exists():
            return c_str
    return str(candidates[-1])   # nothing found yet -- hand back the sensible
                                  # fallback anyway so there's something concrete
                                  # to see and edit in SETTINGS


def default_config():
    """Built fresh (not a static dict) since it probes the real filesystem
    for OS-appropriate locations -- what's "right" depends on this
    machine, not just this platform string."""
    return {
        "retroarch_path": _pick_first_existing(_default_retroarch_candidates()),
        "launch_prefix": "",       # runs before retroarch_path, e.g. "wine" or "WINEPREFIX=... wine"
        "cores_dir": _pick_first_existing(_default_cores_dir_candidates()),
        "core_override": "",       # explicit full path to a core file; wins if set
        "default_format": "dual",
        "tapes_dir": str(BASE_DIR / "tapes"),
        "roms_dir": str(BASE_DIR / "roms"),
        "extra_args": "",          # extra CLI flags inserted before -L <core> <rom>
    }


CONFIG_SECTION = "launchpad"


def load_config():
    cfg = default_config()
    if CONFIG_PATH.exists():
        parser = configparser.ConfigParser()
        try:
            parser.read(CONFIG_PATH)
            if parser.has_section(CONFIG_SECTION):
                for key in cfg:
                    if parser.has_option(CONFIG_SECTION, key):
                        cfg[key] = parser.get(CONFIG_SECTION, key)
        except configparser.Error:
            pass
    return cfg


def save_config(cfg):
    parser = configparser.ConfigParser()
    parser[CONFIG_SECTION] = {k: str(v) for k, v in cfg.items()}
    with open(CONFIG_PATH, "w") as f:
        parser.write(f)


# ---------------------------------------------------------------------------
# Atari 2600 ROM size sanity check (no other consoles supported)
# ---------------------------------------------------------------------------

ATARI2600_SIZES = {2 * 1024, 4 * 1024, 8 * 1024, 12 * 1024, 16 * 1024, 32 * 1024}


def check_atari2600_size(rom_bytes):
    """Returns True if the size matches a standard Atari 2600 cartridge
    size. Not a hard requirement (bankswitched/homebrew ROMs can be other
    sizes) -- just used to print a heads-up if something looks off."""
    return len(rom_bytes) in ATARI2600_SIZES


# ---------------------------------------------------------------------------
# Core discovery
# ---------------------------------------------------------------------------

STELLA_CORE_GLOB = "stella_libretro.*"


_ENV_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def split_command_with_env_prefixes(command_str):
    """Parses a command string that may start with shell-style env var
    assignments, e.g. 'WINEPREFIX=/home/user/.wine-ra WINEDEBUG=-all wine
    /path/to/retroarch.exe'. Returns (env_overrides_dict, remaining_argv).
    subprocess.Popen doesn't understand 'KEY=VAL' as argv[0] the way a
    real shell does, so these need to be pulled out and applied via the
    `env` kwarg instead."""
    tokens = shlex.split(command_str, posix=not sys.platform.startswith("win"))
    env_overrides = {}
    i = 0
    while i < len(tokens) and _ENV_PREFIX_RE.match(tokens[i]):
        key, _, val = tokens[i].partition("=")
        env_overrides[key] = val
        i += 1
    return env_overrides, tokens[i:]


def find_core(cfg):
    """Returns (core_path, message). core_path is None if nothing usable
    was found."""
    override = cfg.get("core_override", "").strip()
    if override:
        if Path(override).exists():
            return override, f"Using core override: {override}"
        return None, f"core_override is set to '{override}' but that file doesn't exist."

    cores_dir = Path(cfg["cores_dir"])
    if not cores_dir.is_dir():
        return None, f"Cores folder not found: {cores_dir}"

    matches = sorted(cores_dir.glob(STELLA_CORE_GLOB))
    if matches:
        return str(matches[0]), f"Found Stella core: {matches[0]}"
    return None, (f"No Stella core (stella_libretro.*) found in {cores_dir}. "
                  f"Place the Atari 2600 core there, or set 'core_override' in SETTINGS.")


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def _require_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as e:
        print("This feature needs the 'sounddevice' package AND the system")
        print("PortAudio library it depends on.")
        print(f"  (import failed: {e})")
        print("Install with:  pip install sounddevice")
        print("If that's already installed, your OS also needs PortAudio itself")
        print("(e.g. 'apt install libportaudio2' on Debian/Ubuntu, or")
        print(" 'brew install portaudio' on macOS).")
        return None


def choose_input_device(sd):
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        print("No input devices found.")
        return None
    print("AVAILABLE INPUT DEVICES:")
    print("(look for your mic/line-in, OR a loopback / \"Monitor of ...\" /")
    print(" \"Stereo Mix\" device if you want to capture system audio output")
    print(" directly, for a fully self-contained test with no physical tape)")
    print()
    default_idx = sd.default.device[0] if sd.default.device[0] is not None else inputs[0][0]
    for i, d in inputs:
        marker = " (default)" if i == default_idx else ""
        print(f"  [{i:2d}] {d['name']}{marker}")
    print()
    choice = ask("DEVICE INDEX (blank = default)", "")
    if choice == "":
        return None
    try:
        return int(choice)
    except ValueError:
        print("Not a number, using default.")
        return None


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def action_listen(cfg, last_loaded):
    sd = _require_sounddevice()
    if sd is None:
        pause()
        return

    device = choose_input_device(sd)

    playback_wav = ask("PATH TO WAV TO AUTO-PLAY FOR SELF-TEST (blank = none)", "")
    playback_samples = None
    if playback_wav:
        try:
            playback_samples = nc.read_wav(playback_wav)
        except (OSError, ValueError) as e:
            print(f"Could not load that WAV: {e}")
            playback_samples = None

    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    try:
        stream = sd.InputStream(samplerate=nc.SAMPLE_RATE, channels=1, dtype="float32",
                                 device=device, callback=callback)
        stream.start()
    except Exception as e:
        print(f"Could not open input device: {e}")
        pause()
        return

    print()
    print("RECORDING STARTED.")
    if playback_samples is not None:
        print(f"Starting test playback of {playback_wav} now on the default output device...")
        sd.play(playback_samples, nc.SAMPLE_RATE)
    print("PRESS ENTER WHEN THE TAPE (OR TEST PLAYBACK) HAS FINISHED.")
    input()

    stream.stop()
    stream.close()
    try:
        sd.stop()   # stop any lingering playback
    except Exception:
        pass

    if not chunks:
        print("No audio was captured.")
        pause()
        return

    audio = np.concatenate(chunks).flatten()
    print(f"Captured {len(audio) / nc.SAMPLE_RATE:.1f}s of audio. Decoding...")
    _decode_and_save(audio, cfg, last_loaded)
    pause()


def action_load_wav(cfg, last_loaded):
    path = ask("PATH TO WAV FILE")
    if not path:
        return
    try:
        audio = nc.read_wav(path)
    except (OSError, ValueError) as e:
        print(f"Could not load that WAV: {e}")
        pause()
        return
    print(f"Loaded {len(audio) / nc.SAMPLE_RATE:.1f}s of audio. Decoding...")
    _decode_and_save(audio, cfg, last_loaded)
    pause()


def _decode_and_save(audio, cfg, last_loaded):
    try:
        rom_bytes = nc.audio_to_rom(audio, verbose=True)
    except ValueError as e:
        print(f"DECODE FAILED: {e}")
        return

    if not check_atari2600_size(rom_bytes):
        print(f"NOTE: {len(rom_bytes)} bytes isn't a standard Atari 2600 cartridge "
              f"size (2/4/8/12/16/32 KB). Could still be a valid bankswitched/homebrew "
              f"ROM -- saving it anyway.")

    roms_dir = Path(cfg["roms_dir"])
    roms_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"loaded_{time.strftime('%Y%m%d_%H%M%S')}.a26"
    out_name = ask("SAVE ROM AS", default_name)
    out_path = roms_dir / out_name
    out_path.write_bytes(rom_bytes)
    print(f"Saved: {out_path}")

    last_loaded["path"] = out_path


def action_save_to_tape(cfg, last_loaded):
    rom_path = ask("PATH TO ROM FILE TO ENCODE")
    if not rom_path or not Path(rom_path).exists():
        print("File not found.")
        pause()
        return
    rom_bytes = Path(rom_path).read_bytes()

    fmt = ask("FORMAT (classic/dual/fast)", cfg["default_format"])
    if fmt not in nc.PROFILES:
        print(f"Unknown format '{fmt}', using '{cfg['default_format']}'.")
        fmt = cfg["default_format"]

    print("Encoding...")
    samples = nc.rom_to_audio(rom_bytes, format_name=fmt, amplitude=0.8)
    duration = len(samples) / nc.SAMPLE_RATE
    print(f"Encoded {len(rom_bytes)} bytes to {duration:.1f}s of '{fmt}' audio.")

    # Always write a real WAV to disk, per the testing workflow: this file
    # can be played back through system audio later (option 4) to test
    # LISTEN, or fed to a real tape recorder.
    tapes_dir = Path(cfg["tapes_dir"])
    tapes_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(rom_path).stem
    wav_name = f"{base_name}_{fmt}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    wav_path = tapes_dir / wav_name
    nc.write_wav(wav_path, samples)
    print(f"Saved tape WAV: {wav_path}")

    if ask("VERIFY ROUND-TRIP IN MEMORY NOW? (y/n)", "y").lower().startswith("y"):
        try:
            decoded = nc.audio_to_rom(samples, verbose=False)
            print("VERIFY OK: matches original exactly." if decoded == rom_bytes
                  else "VERIFY FAILED: decoded bytes differ!")
        except ValueError as e:
            print(f"VERIFY FAILED: {e}")

    if ask("PLAY IT THROUGH SPEAKERS NOW? (y/n)", "n").lower().startswith("y"):
        _play_wav_file(wav_path)

    pause()


def _play_wav_file(path):
    sd = _require_sounddevice()
    if sd is None:
        return
    try:
        samples = nc.read_wav(path)
    except (OSError, ValueError) as e:
        print(f"Could not load that WAV: {e}")
        return
    print(f"About to play {len(samples) / nc.SAMPLE_RATE:.1f}s of audio.")
    input("Start your tape recorder (or arm LISTEN in another terminal), then press ENTER.")
    sd.play(samples, nc.SAMPLE_RATE, blocking=True)
    print("Playback finished.")


def action_play_wav(cfg, last_loaded):
    path = ask("PATH TO WAV FILE TO PLAY")
    if not path or not Path(path).exists():
        print("File not found.")
        pause()
        return
    _play_wav_file(path)
    pause()


def action_launch_retroarch(cfg, last_loaded):
    rom_path = last_loaded["path"]
    if rom_path is None:
        typed = ask("NO ROM LOADED THIS SESSION. PATH TO ROM FILE", "")
        if not typed:
            return
        rom_path = Path(typed)
        if not rom_path.exists():
            print("File not found.")
            pause()
            return
        last_loaded["path"] = rom_path

    core_path, message = find_core(cfg)
    print(message)
    if core_path is None:
        pause()
        return

    full_command = f"{cfg.get('launch_prefix', '').strip()} {cfg['retroarch_path']}".strip()
    env_overrides, retroarch_cmd = split_command_with_env_prefixes(full_command)
    if not retroarch_cmd:
        print("RetroArch path/command is empty (or only contains environment "
              "variable prefixes). Set it in SETTINGS.")
        pause()
        return

    prefix_set = bool(cfg.get("launch_prefix", "").strip())

    resolved = shutil.which(retroarch_cmd[0])
    if resolved is None:
        # Not found on PATH -- but it might still be a direct path.
        candidate = Path(retroarch_cmd[0])
        if not candidate.exists():
            if prefix_set and retroarch_cmd[0] in cfg["launch_prefix"]:
                print(f"Can't find the command prefix '{retroarch_cmd[0]}' (from your")
                print("Command prefix setting). If you're running RetroArch natively")
                print("and don't actually need a wrapper like Wine, clear that setting")
                print("in SETTINGS option 2 (type <CLEAR>) instead of fixing this path.")
            else:
                print(f"Can't find RetroArch at '{retroarch_cmd[0]}'.")
                print("Set the correct path (or command, e.g. 'flatpak run org.libretro.RetroArch') "
                      "in SETTINGS.")
            pause()
            return
        if candidate.is_file() and not os.access(candidate, os.X_OK):
            print(f"Found '{candidate}' but it isn't marked executable, which is exactly")
            print("what a bare 'Permission denied' on launch means. Fix it with:")
            print(f'    chmod +x "{candidate}"')
            print("(This is common right after downloading/extracting a portable build")
            print(" or an AppImage -- the execute bit doesn't survive a zip/tar extract.)")
            pause()
            return

    extra_args = shlex.split(cfg.get("extra_args", ""), posix=not sys.platform.startswith("win"))

    cmd = retroarch_cmd + extra_args + ["-L", core_path, str(rom_path)]
    env_prefix_str = "".join(f"{k}={v} " for k, v in env_overrides.items())
    print("LAUNCHING: " + env_prefix_str + " ".join(cmd))
    launch_env = os.environ.copy()
    launch_env.update(env_overrides)
    try:
        proc = subprocess.Popen(cmd, env=launch_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except PermissionError as e:
        print(f"Permission denied launching RetroArch: {e}")
        print("Check that the RetroArch executable itself has the execute bit set")
        print(f'(chmod +x "{retroarch_cmd[0]}") and that this ROM file and the core')
        print("file are both readable by your user.")
        pause()
        return
    except OSError as e:
        print(f"Failed to launch RetroArch: {e}")
        pause()
        return

    # Popen succeeding just means the process started -- it says nothing
    # about whether RetroArch/Wine then immediately crashed or exited on
    # its own. Give it a moment and actually check, since that's exactly
    # the "no errors, but nothing happens" symptom a silent crash produces.
    print("Waiting a moment to confirm it's actually still running...")
    time.sleep(2.0)
    returncode = proc.poll()
    if returncode is None:
        print("RetroArch launched and is still running.")
    else:
        print(f"RetroArch exited immediately (exit code {returncode}) -- it did NOT stay running.")
        try:
            out, err = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        if err and err.strip():
            print("--- stderr (most recent output) ---")
            print(err.strip()[-2000:])
        if out and out.strip():
            print("--- stdout (most recent output) ---")
            print(out.strip()[-2000:])
        if not (err and err.strip()) and not (out and out.strip()):
            print("(it produced no output at all, which usually points to a wrapper/")
            print(" prefix command problem rather than RetroArch itself -- see below)")
        print()
        print("Common causes: a core built for the wrong CPU architecture or a")
        print("RetroArch ABI version this build doesn't match, a missing shared")
        print("library dependency the core itself needs (check with `ldd` on the")
        print("core file), an invalid/corrupt ROM for that core, or a permissions")
        print("issue on the core file. Try running the exact LAUNCHING command")
        print("above directly in a terminal (outside this script) to see the full")
        print("output live -- that alone often tells you exactly what's wrong.")
        if cfg.get("launch_prefix", "").strip():
            print()
            print(f"NOTE: Command prefix is currently set to '{cfg['launch_prefix']}'.")
            print("If you're running RetroArch natively (no Wine/wrapper needed),")
            print("that prefix could be exactly what's breaking this -- clear it in")
            print("SETTINGS option 2 (type <CLEAR>) and try again.")
    pause()


def action_settings(cfg, last_loaded):
    while True:
        cls()
        hr("=")
        center("SETTINGS")
        hr("=")
        print()
        print(f"  1) RetroArch path       : {cfg['retroarch_path']}")
        print(f"  2) Command prefix (opt.): {cfg['launch_prefix'] or '(none)'}")
        print(f"  3) Cores folder         : {cfg['cores_dir']}")
        print(f"  4) Core override (opt.) : {cfg['core_override'] or '(not set -- auto-detect in cores folder)'}")
        print(f"  5) Default tape format  : {cfg['default_format']}")
        print(f"  6) Tapes output folder  : {cfg['tapes_dir']}")
        print(f"  7) ROMs output folder   : {cfg['roms_dir']}")
        print(f"  8) Extra RetroArch args : {cfg['extra_args'] or '(none)'}")
        print(f"  9) BACK")
        print()
        print(f"(settings are saved to {CONFIG_PATH})")
        print()
        choice = ask("SELECT", "9")
        if choice == "1":
            print("Can include leading VAR=value environment prefixes, e.g.")
            print("'WINEPREFIX=/home/user/.wine-ra WINEDEBUG=-all wine /path/to/retroarch.exe',")
            print("or use option 2 below to keep a wrapper command separate from this path.")
            cfg["retroarch_path"] = ask("RetroArch path/command", cfg["retroarch_path"])
        elif choice == "2":
            print("Runs immediately before the RetroArch path, e.g. 'wine', 'gamemoderun',")
            print("'prime-run', or env var assignments like 'WINEPREFIX=/home/user/.wine-ra'.")
            print("Combined as: <this prefix> <RetroArch path> -L <core> <rom>")
            cfg["launch_prefix"] = ask_clearable("Command prefix", cfg["launch_prefix"])
        elif choice == "3":
            cfg["cores_dir"] = ask("Cores folder path", cfg["cores_dir"])
        elif choice == "4":
            cfg["core_override"] = ask_clearable("Full path to a specific core file",
                                                  cfg["core_override"])
        elif choice == "5":
            fmt = ask("Default format (classic/dual/fast)", cfg["default_format"])
            if fmt in nc.PROFILES:
                cfg["default_format"] = fmt
            else:
                print("Unknown format, ignored.")
                time.sleep(1)
        elif choice == "6":
            cfg["tapes_dir"] = ask("Tapes output folder", cfg["tapes_dir"])
        elif choice == "7":
            cfg["roms_dir"] = ask("ROMs output folder", cfg["roms_dir"])
        elif choice == "8":
            print("These get inserted before '-L <core> <rom>', e.g. '-f --verbose'")
            print("or '--appendconfig /path/to/override.cfg'.")
            cfg["extra_args"] = ask_clearable("Extra RetroArch CLI args", cfg["extra_args"])
        else:
            save_config(cfg)
            return
        save_config(cfg)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # The launchpad always runs from its own install location (the
    # RetroArch directory it was dropped into), regardless of the shell's
    # current directory it was invoked from.
    os.chdir(BASE_DIR)

    cfg = load_config()
    if not CONFIG_PATH.exists():
        save_config(cfg)   # write out defaults on first run
    last_loaded = {"path": None}

    theme_on()
    try:
        boot_screen()
        input("PRESS ENTER TO CONTINUE" + BLINK_CURSOR())

        while True:
            draw_menu(cfg, last_loaded)
            choice = ask("SELECT", "")
            if choice == "1":
                action_listen(cfg, last_loaded)
            elif choice == "2":
                action_load_wav(cfg, last_loaded)
            elif choice == "3":
                action_save_to_tape(cfg, last_loaded)
            elif choice == "4":
                action_play_wav(cfg, last_loaded)
            elif choice == "5":
                action_launch_retroarch(cfg, last_loaded)
            elif choice == "6":
                action_settings(cfg, last_loaded)
            elif choice == "7":
                break
    except KeyboardInterrupt:
        pass
    finally:
        cls()
        center("BYE.")
        theme_off()


if __name__ == "__main__":
    main()
