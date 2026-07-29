Use nes_cassette.py and the latest retro_launchpad.py together in your Retroarch directory. Linux users installing retroarch via flatpak don't need to worry about where it's installed, only that both .py files are together in a folder. Use the settings menu to point to the correct folders for stuff. Make sure you have your cores file path
correct, and that it's looking for the correct one!

GOAL: Emulation front end that mimics a C64 and uses consumer-level tape decks to load roms into an existing emulator.

SO FAR: There are 3 methods to encode the rom into an audio file to be saved to a cassette tape. The 3 methods are:
1) Standard: safest bet, but slowest. Accounts for most audio glitches that come from cassette interfaces.
2) Dual: still safe, but slightly increases bandwidth by using dual bit tones.
3) Fast: less safe, but increases bandwidth by quite a bit.

*Need to test various tape deck hardware with these settings
