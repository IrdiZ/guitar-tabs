# 🎸 Guitar Tab Generator

AI-powered tool to generate guitar tablature from audio files or YouTube URLs.

Built for songs with no existing tabs online (like obscure Albanian music).

## Features

- **Audio → Tabs**: Upload any audio file (mp3, wav, etc.)
- **YouTube Support**: Paste a YouTube URL, get tabs
- **Smart Fret Mapping**: AI chooses playable positions
- **ASCII Output**: Standard guitar tab format

## Quick Start

```bash
# Install dependencies
pip install librosa numpy soundfile yt-dlp

# From local file
python guitar_tabs.py song.mp3 -o tabs.txt

# From YouTube
python guitar_tabs.py "https://youtube.com/watch?v=..." -o tabs.txt

# Lower confidence threshold for noisy audio
python guitar_tabs.py song.mp3 -c 0.1
```

## How It Works

1. **Pitch Detection**: Uses librosa to analyze audio frequencies
2. **Note Onset Detection**: Identifies when notes start
3. **MIDI Conversion**: Converts frequencies to musical notes
4. **Fret Mapping**: Maps notes to guitar positions considering:
   - Playability (prefers lower frets)
   - Hand continuity (minimizes jumps)
   - String preference (middle strings easier)

## Output Example

```
e|----------------|
B|----------------|
G|----2-------0---|
D|--------0-------|
A|----------------|
E|----------------|
```

## Limitations

- Works best with clean guitar audio (no vocals/drums)
- Use [Moises.ai](https://moises.ai) to extract guitar stem first
- Polyphonic detection (chords) is experimental

## Tech Stack

- Python 3.11+
- librosa (audio analysis)
- numpy (signal processing)
- yt-dlp (YouTube download)

## License

MIT

## Contributing

PRs welcome! Ideas:
- [ ] Guitar Pro export (.gp5)
- [ ] Chord detection
- [ ] Web UI
- [ ] Better polyphonic support
