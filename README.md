# Gesture-Speech Alignment

## Why can't you stop watching a speaker's hands?

You do it without noticing. And many of us wave our hands on phone calls where nobody can see them. Some linguists think that's because speech and gesture aren't two channels but one system in the brain, so the hand and the voice should hit emphasis *at the same moment*.

I tested that on 8 TED talks: 231 hand movements against every moment of vocal stress.

**So far the answer is "can't tell yet": with 8 clips, the hands landed no closer to the stress than random timing would (z = 0.13, p = 0.45), but the data is too small to settle it.** A small real link could easily be hiding in 231 gestures. What this repo delivers is the groundwork: a test that was checked on planted data, a pipeline that handles slides and missed hands, and a clear target of how much more data is needed. It also corrects my own first version, which "found" a perfect 100% match that turned out to be meaningless.

## What this project explores

- **Statistical analysis:** how to test a human-behavior claim properly, and how an easy-looking metric can mislead.
- **Multimodal communication:** whether the voice and the hands carry emphasis together.
- **Human psychology:** whether speech and gesture form one integrated system, as David McNeill argued.

## The question

When a speaker stresses a word with their voice, does a hand gesture land at the same moment?

## How it works

1. **Voice:** pitch and loudness are extracted (Praat via Parselmouth). A stress point is a pitch peak that coincides with a loudness peak, about one per second.
2. **Hands:** MediaPipe tracks the wrists. A gesture is an unusually fast hand movement for that speaker.
3. **Test:** for each gesture, measure the distance to the nearest stress point. Compare with what random timing gives (the stress timeline is slid by a random amount 2000 times). If real gestures sit closer than almost all random slides, there is a link.
4. **Only when the speaker is on screen:** slides and cutaways are removed, since no gesture can be seen there.

## Results

| Clip | Gestures | z | p |
|---|---|---|---|
| Clay Shirky | 31 | 0.05 | 0.51 |
| Massimo Banzi | 23 | -1.66 | 0.95 |
| Jennifer Pahlka | 37 | -1.70 | 0.97 |
| Jimmy Wales | 23 | 0.76 | 0.25 |
| Luis von Ahn | 25 | 1.70 | 0.027 |
| Amanda Palmer | 25 | 0.13 | 0.48 |
| Linus Torvalds | 47 | -0.23 | 0.64 |
| Tim Berners-Lee | 20 | 1.33 | 0.10 |

z above 0 means gestures are closer to stress than chance; p near 0.5 means it looks like luck.

- **All clips together:** z = 0.13, p = 0.45. Gestures are on average 0.58 s from stress, against 0.60 s for chance.
- **Effect size:** gestures are 2% closer than chance, with a 95% confidence interval from 11% farther to 15% closer. That includes zero.
- **One clip looks significant (von Ahn, p = 0.027).** After correcting for testing 8 clips, p = 0.22, so it does not count.
- **Control:** pairing gestures with a different clip's voice gives z about 0, so the test is not producing false signals.
- Full numbers: `results/v2_per_clip.csv` and `results/stats_summary.txt`.

## What we tried

- **The first version reported a 100% match rate and was meaningless.** Stress was marked about 4 times per second (roughly every syllable) with a wide time window, so almost any gesture landed near one. A simulation showed that test had zero power: it missed a planted link even when 60% of gestures were locked to stress.
- **A new test replaced it.** It uses a continuous distance with no window, and was checked on simulated data (about 5% false positives, and it detects planted links).
- **Stress was made selective** (pitch and loudness peaking together), matching the hand side, which only fires on visible movement.
- **Slides were removed.** The speaker was on screen only 13–72% of the time in the first pass.
- **Hand detection was improved**, raising gestures from 109 to 231.

Each fix made the method more trustworthy. None produced a significant link, and the earlier weak trend (z about 1.4) disappeared when gestures were detected properly.

## The limit: not enough data

- **231 gestures across 8 clips** gives enough power to detect a fairly strong link (about 0.9 z per clip), not a small one.
- The confidence interval is wide, so a small real effect could be hiding in it. More clips would narrow it.
- This does not show more data will find an effect. It shows only that this data cannot settle the question.

## Other limits

- The detector fires on any fast wrist movement, not only beat gestures.
- Stress is a pitch-and-loudness proxy, with no human labeling.
- Speaker visibility comes from face and hand detection, which is imperfect.
- TED speakers are rehearsed, so results may not carry over to casual speech.

## Next steps

More clips and speakers, a human-labeled subset to check both detectors, and a comparison of rehearsed and spontaneous speech.

## Run it

```
python src/validate_simulation.py   # test sanity check (no video)
python src/extract_tracks.py        # slow, once: hand tracks for every clip
python src/run_v2_analysis.py       # per-clip results -> results/v2_per_clip.csv
python src/stats_summary.py         # group statistics -> results/stats_summary.txt
```

Clips go in `data/clips/` with matching audio in `data/tmp/` (see `src/download_clips.py` and `src/audio_prosody.py`).

## Files

- `src/audio_prosody.py`: pitch and loudness, stress points
- `src/gesture_tracking.py`: hand tracking and gesture points
- `src/visibility.py`: when the speaker is on screen
- `src/alignment.py`: distance test and permutation null
- `src/validate_simulation.py`: planted-link check of the test
- `src/run_v2_analysis.py`, `src/stats_summary.py`: final analysis
- `results/v2_per_clip.csv`, `results/stats_summary.txt`: final numbers
- `results/per_clip_summary.csv`: the first, superseded match-rate analysis (kept for reference)

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
