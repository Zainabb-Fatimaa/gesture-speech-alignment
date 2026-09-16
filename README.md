# Gesture-Speech Alignment

A small pilot pipeline testing whether hand gestures (beat gestures) time-align with vocal stress in natural speech. Built out of an interest in analyzing human patterns, using computer vision and audio signal processing as instruments to ask a genuinely scientific question rather than to build a product.

## The question

When a speaker emphasizes a word with their voice, does their hand gesture shift with it? Linguists such as David McNeill have argued gesture and speech form one integrated system rather than two separate channels. This project builds a small, reproducible test of that idea on real talk footage.

## How it works

Two signals are extracted in parallel for each clip, then compared.

1. **Prosody (audio side)**, `audio_prosody.py`: extracts pitch (F0) and intensity contours with Parselmouth (a Python wrapper around Praat), then detects local peaks within continuous voiced speech as candidate stress points.
2. **Gesture (video side)**, `gesture_tracking.py`: tracks hand position frame by frame with MediaPipe's HandLandmarker, computes vertical velocity, and detects local peaks as candidate gesture points, mostly beat gestures.
3. **Alignment**, `alignment.py`: for every gesture peak, finds the nearest stress peak within a matching window (default 0.5s), records the timing offset, and runs a permutation test that circularly shifts the stress-peak timeline hundreds of times to build a chance-level baseline for comparison.

## Data

8 clips (30 to 90 seconds each) trimmed from TED's "Open Source, Open World" playlist: single speaker, waist-up framing, natural unscripted delivery, clean audio. Clip list is the `clip` column in `per_clip_summary.csv`.

## Results

Full per-clip numbers are in `per_clip_summary.csv` (attached). Summary below.

| Clip | Gesture peaks | Match rate | Chance match rate | p-value | Median offset (ms) | Gesture precedes stress |
|---|---|---|---|---|---|---|
| Clay Shirky | 22 | 100% | 100% | 1.00 | -25 | 59% |
| Massimo Banzi | 3 | 100% | 100% | 1.00 | -50 | 67% |
| Jennifer Pahlka | 19 | 100% | 100% | 1.00 | -18 | 58% |
| Jimmy Wales | 1 | 100% | 100% | 1.00 | +197 | 0% |
| Luis von Ahn | 21 | 100% | 99% | 0.88 | +17 | 43% |
| Linus Torvalds | 17 | 100% | 99% | 0.83 | +27 | 41% |
| Amanda Palmer | 15 | 100% | 97% | 0.71 | -10 | 53% |
| Tim Berners-Lee | 11 | 100% | 100% | 1.00 | +10 | 45% |

## Reading these results honestly

The raw match rate is 100% on every clip. On its own that number is not meaningful: the permutation test shows the chance-level match rate is also close to 100% on most clips, and p-values are all well above 0.05. In natural speech, stress peaks (peaks in pitch or loudness) occur frequently enough that almost any gesture timing lands near one by coincidence, given the current 0.5 second matching window. So this pilot does not currently show statistically significant gesture-stress synchrony beyond chance, and that should not be overstated in any writeup.

What does hold up as a smaller, honest signal is the offset direction: in 6 of 8 clips, gestures tend to arrive before the stressed syllable more often than after it (negative median offset, over 50% precede), which is directionally consistent with the gesture-speech literature. That is the finding worth reporting, framed as a tendency, not as a proven alignment effect.

Separately, gesture peak detection itself (does the hand-tracking step correctly flag real hand movement as a peak when actual vocal stress was present) was manually spot-checked against the source video and held up in the 75 to 95% range depending on clip and speaker expressiveness. This is a detector-accuracy number, kept deliberately separate from the alignment claim above.

## Files

- `gesture_tracking.py`, hand tracking and gesture peak detection
- `audio_prosody.py`, pitch and intensity extraction, stress peak detection (fixed version: peaks are only searched for within continuous voiced audio, not across zero-filled silent gaps, which was inflating stress peak counts in an earlier version)
- `alignment.py`, peak matching, permutation test, plotting
- `download_clips.py`, fetches and trims source clips
- `analysis.ipynb` / `analysis_local.ipynb`, end-to-end run and figures
- `per_clip_summary.csv`, per-clip results (attached separately below)
- `*_timeline.png`, per-clip gesture vs stress event timelines
- `offset_histogram.png`, distribution of gesture-stress timing offsets across all matched peaks

## Requirements

```
mediapipe
opencv-python
praat-parselmouth
scipy
numpy
matplotlib
pandas
yt-dlp
tqdm
```

## Limitations and next steps

This is a pilot, not a validated study: 8 clips, no manual ground-truth annotation of gesture or stress events, no inter-rater reliability check. Planned next steps: a narrower, principled matching window instead of a fixed 0.5 seconds, a larger and more diverse clip set across speakers and languages, and manual annotation on a subset to validate the automatic detectors against human judgment.