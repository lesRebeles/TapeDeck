Use nes_cassette.py and the latest retro_launchpad.py together in your Retroarch directory. Linux users installing retroarch via flatpak don't need to worry about where it's installed, only that both .py files are together in a folder. Use the settings menu to point to the correct folders for stuff. Make sure you have your cores file path
correct, and that it's looking for the correct one!

GOAL: Emulation front end that mimics a C64 and uses consumer-level tape decks to load roms into an existing emulator.

SO FAR: There are 3 methods to encode the rom into an audio file to be saved to a cassette tape. The 3 methods are:
1) Standard: safest bet, but slowest. Accounts for most audio glitches that come from cassette interfaces.
2) Dual: still safe, but slightly increases bandwidth by using dual bit tones.
3) Fast: less safe, but increases bandwidth by quite a bit.

*Need to test various tape deck hardware with these settings

This is a VERY early version, and I have not tested this with physical hardware yet (an actual tape deck), but with the error correction in the scripts, I think this could be cool. I want to dive deeper into the recording/playback modes and make the most of the tape's bandwidth, increasing speed and retaining accuracy as I go... decreasing loading times. Currently, it's utilizing Retroarch for its Atari 2600 emulation core. Yes, I know, the Atari 2600 didn't use cassettes, but the file size is perfect for this experimental implementation. Projected load time for a typical 2600 rom from a tape is approximately 1:10. -LR
