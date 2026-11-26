# Bodh (बोध)

Bodh is a local research toolkit for audio authenticity and human perception. It studies how a recording was produced, how listeners perceive it, and which recording properties can mislead a detector.

Bodh runs on a CPU and uses local files. The pipeline includes audio feature extraction, dataset provenance checks, baseline detectors, a small spectrogram Transformer trained from scratch, and evaluation against observed listener ratings.

## What the experiments found

- **Recording differences can look like detection signal.** In a 22-clip speech pilot, the median quiet-frame gap between genuine and spoofed clips fell from 21.3 to 3.7 percentage points after excluding quiet frames at the clip edges.
- **Official speaker IDs can miss overlap.** The ASVspoof source map exposed 10 shared underlying speakers and three reused source recordings in the sampled train and development partitions. The stricter split recorded and excluded the overlapping clips.
- **The lightweight baseline held up best.** On 200 audited ASVspoof evaluation clips after quiet-edge trimming, simple cues reached **0.763 ROC-AUC**, compared with **0.727** for a randomly initialized Transformer and **0.714** for the pretrained Transformer. The pretrained embeddings also did not improve on the training-mean predictor for held-out Lavan rating groups.

![Aggregate results from Lavan speaker pairs and the quiet-frame comparison across both audio pilots](docs/figures/pilot-insights.png)

See the [research notes](docs/research.md) for datasets and pilot findings, the [evaluation report](docs/evaluation.md) for the model comparisons, and the [media review guide](docs/media-review.md) for the independent audio benchmark and claim-evidence demo.

## Quickstart

Use Python 3.12 with FFmpeg and FFprobe on your PATH. On macOS, FFmpeg can be installed with Homebrew. On Ubuntu, it is available as the `ffmpeg` package.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
python scripts/signal_fixture.py data/tones
bodh eda data/tones/manifest.csv data/tones runs/tones
```

The example generates eight test tones locally and produces clip features, a JSON summary, and plots in `runs/tones/`. It runs offline after installation and uses no dataset download. Choose fresh output directories when rerunning a command.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`src/bodh/`](src/bodh/) | CLI, audio processing, manifests, listener statistics, baselines, and the Transformer |
| [`scripts/`](scripts/) | Dataset acquisition, split audits, experiment runners, and generated test audio |
| [`tests/`](tests/) | Tests that run on generated audio and synthetic rating records |
| [`docs/`](docs/) | Study design, aggregate results, and reproduction commands |

The [reproduction guide](docs/reproduction.md) covers the public datasets and full experiment sequence. Transformer pretraining and transfer use the optional `pretrain` extra. Run the local tests with `python -m unittest discover -s tests -v`.

## Data and scope

Raw audio, speaker and listener records, local results, and model checkpoints stay in ignored directories. The repository contains code, methods, and aggregate findings. Source datasets have their own terms and should be acquired from their publishers.

A dataset's authenticity label describes how a recording was produced. Listener ratings describe judgments in a particular study. Truthfulness requires independent evidence. The current results cover bounded research samples and leave cross-corpus performance unmeasured.
