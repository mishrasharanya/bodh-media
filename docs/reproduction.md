# Reproduce Bodh's audio experiments

Run commands from the repository root with Python 3.12, FFmpeg and FFprobe. The dataset downloaders also use curl. The [quickstart](../README.md#quickstart) installs the core package and tests the pipeline with generated audio. For the Transformer experiments, install the optional dependencies.

```sh
python -m pip install -e '.[pretrain]'
export OPENBLAS_NUM_THREADS=1
```

The tested research runs used Python 3.12.2 on an Apple M3 Pro, FFmpeg 9.0.2, and PyTorch 2.2.2 on CPU. Each command below writes to a new directory. Use fresh output paths for reruns. Keep publisher audio, raw rating tables, checkpoints, and per-clip outputs under the ignored `data/` and `runs/` directories.

## Speech and listener pilots

```sh
python scripts/public_demo.py data/behavioral-signals-demo --download
bodh eda data/behavioral-signals-demo/manifest.csv data/behavioral-signals-demo runs/speech-pilot

python scripts/lavan2025.py data/lavan2025 data/lavan2025/experiment1b --download
bodh eda data/lavan2025/experiment1b/manifest.csv data/lavan2025 runs/lavan-1b
bodh perception data/lavan2025/experiment1b/manifest.csv \
  data/lavan2025/experiment1b/trials.csv > runs/lavan-1b/listener_report.json

python scripts/pilot_insights.py runs/pilot-insights
```

The Lavan importer downloads the two Experiment 1b rating tables and the 80 corresponding WAVs. It verifies the source listing and file hashes against the study snapshot and assigns local pseudonyms for listeners. The speech demo is pinned to its publisher's dataset revision. [Lavan OSF project](https://osf.io/mrw5q/) · [Behavioral Signals dataset](https://huggingface.co/datasets/behavioralsignals/deepfake-detection-demo)

For the Belin rating reanalysis, save the supplements under `data/belin2017/`.

- Save the experimental audio as `trustworthiness.wav`. [S1 Audio](https://doi.org/10.1371/journal.pone.0185651.s001)
- Save the control audio as `control.wav`. [S2 Audio](https://doi.org/10.1371/journal.pone.0185651.s002)
- Save the ratings archive as `ratings.zip`. [S1 Data](https://doi.org/10.1371/journal.pone.0185651.s003)

```sh
python scripts/belin2017.py data/belin2017 runs/belin2017
bodh eda runs/belin2017/montage_manifest.csv data/belin2017 runs/belin2017/audio
```

## ASVspoof acquisition and overlap audit

```sh
python scripts/fetch_asvspoof2019.py data/asvspoof2019-pilot --seed 2026

mkdir -p data/asvspoof2019-reference
curl -fL https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz \
  -o data/asvspoof2019-reference/source-maps.tar.gz
tar -xzf data/asvspoof2019-reference/source-maps.tar.gz -C data/asvspoof2019-reference \
  ASVspoof2019_LA_VCTK_MetaInfo.tsv

python scripts/audit_asvspoof2019.py data/asvspoof2019-pilot \
  data/asvspoof2019-reference/ASVspoof2019_LA_VCTK_MetaInfo.tsv \
  data/lavan2025/experiment1b/manifest.csv runs/asvspoof-preflight
```

The downloader fetches selected ranges of the official ZIP. The audit checks the train, development and evaluation audio against the challenge labels and source map and records all exclusions. It also exports the Lavan subset whose documented speakers are absent from ASVspoof train and development. [Dataset release](https://doi.org/10.7488/ds/2555) · [Organizers' source maps](https://www.asvspoof.org/index2019.html)

## Baselines and Transformer

```sh
python scripts/asvspoof_baseline.py train data/asvspoof2019-pilot runs/asvspoof-preflight runs/asvspoof-baseline
python scripts/asvspoof_baseline.py evaluate data/asvspoof2019-pilot runs/asvspoof-preflight runs/asvspoof-baseline

python scripts/asvspoof_pretrain.py data/asvspoof2019-pilot runs/asvspoof-preflight runs/asvspoof-pretrain

python scripts/asvspoof_transfer.py fit data/asvspoof2019-pilot \
  runs/asvspoof-preflight runs/asvspoof-pretrain/encoder.npz runs/asvspoof-transfer
python scripts/asvspoof_transfer.py evaluate data/asvspoof2019-pilot \
  runs/asvspoof-preflight runs/asvspoof-pretrain/encoder.npz runs/asvspoof-transfer
python scripts/lavan_transfer.py data/lavan2025 \
  runs/asvspoof-preflight runs/asvspoof-pretrain/encoder.npz runs/lavan-transfer
```

The encoder has four bidirectional Transformer blocks with four attention heads, hidden width 128 and feed-forward width 256. It uses 40-band log-mel patches that span eight frames. The pretraining objective reconstructs masked patches and ignores padding. Training starts with random weights and saves local checkpoints as NumPy archives with tensor arrays and JSON metadata. The [evaluation report](evaluation.md) describes the downstream comparisons.

## Use your own local corpus

An asset manifest has the header `dataset,asset_id,modality,path,split,source_group,condition,speaker_id,generator_id`. Paths are relative to the audio root. Use `source_group` for a known shared parent recording or matched stimulus group. Use `unsplit` for descriptive studies. Speaker and generator IDs may be blank.

A trial manifest has the header `dataset,asset_id,study_arm,participant_id,endpoint,value`. Use pseudonymous participant IDs. Document each endpoint's scale and meaning. The perception command joins by `(dataset, asset_id)` and computes equal-stimulus-weight summaries.

```sh
bodh eda path/to/manifest.csv path/to/audio-directory runs/custom-audio
bodh perception path/to/manifest.csv path/to/trials.csv
```

The audio pipeline retains source format details, averages channels to mono, and resamples to 16 kHz. It uses 25 ms Hann windows with a 10 ms hop and a 512-point FFT. The pooled log-mel summary preserves level differences. The resampled view omits information above 8 kHz and downmixing can cancel out-of-phase content. Keep the native recording when its terms permit it. Review `errors.csv` after a run. A failed clip yields a nonzero exit code and a partial report.
