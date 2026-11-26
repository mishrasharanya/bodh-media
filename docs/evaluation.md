# ASVspoof 2019 evaluation pilot

Bodh's first authenticity evaluation uses a local sample of the official ASVspoof 2019 Logical Access data. It is designed to check the pipeline and compare small models under documented split rules. The corpus uses older English-language synthesis and voice conversion systems. Results on this pilot will describe these clips and conditions. [Dataset release](https://doi.org/10.7488/ds/2555) · [Corpus paper](https://www.eurecom.eu/publication/6352/download/sec-publi-6352.pdf)

## Acquisition and split checks

1. **Download a fixed sample.** Fetch the selected sections of Edinburgh's official ZIP and verify the archive version, byte offsets and ZIP checksums. Use seed 2026 to take 100 bona fide and 100 spoof clips from each official partition. Cycle across speakers for bona fide audio and attack IDs for spoof audio. The 600 clips and protocol files occupy about 39 MiB locally. The balanced sample supports comparisons under controlled conditions.
2. **Audit the official sample.** Check every label and official speaker ID against the challenge protocols. Decode every FLAC and hash both the raw file and the standardized waveform. Use the organizers' VCTK map to look for reused source recordings and speakers, including the input speakers for voice conversion. The sampled train and development partitions shared **10 underlying speakers and 3 source recordings**, even though their official speaker IDs and file hashes did not overlap. [Organizer file maps](https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz)
3. **Freeze the stricter split.** Keep train membership, then exclude any development or evaluation clip that shares an audited speaker, source recording or exact hash with a previously accepted split. Keep the original manifests and record every exclusion separately. The final split contains **200 train, 116 development and 200 evaluation clips**. The 84 exclusions are all from development, leaving 50 bona fide and 66 spoof clips. The strict sample has zero measured overlap in the audited source and audio keys. The checks cover the published source map and exact hashes.
4. **Label test conditions before fitting.** The 200 evaluation clips contain **100 bona fide, 15 from previously seen algorithms and 85 from unseen algorithms**. The published algorithm equivalents A04/A16 and A06/A19 count as seen. Development retains four of the six training algorithms after overlap filtering, so its calibration coverage is narrower. [Corpus paper, Table 1](https://www.eurecom.eu/publication/6352/download/sec-publi-6352.pdf)
5. **Check the listener corpus separately.** Of the 40 Lavan speaker pairs, 18 share a mapped speaker with the ASVspoof train or development sets. The exported sensitivity manifest keeps the other 22 pairs. These stimuli were already inspected and share the VCTK source corpus with ASVspoof. The slice supports a sensitivity analysis. An external holdout requires an independently sourced corpus.

The audit records split hashes, the source-map hash, overlap counts, and evaluation algorithm categories in `runs/asvspoof-preflight/`. Evaluation audio was decoded for integrity and overlap checks. See the [reproduction guide](reproduction.md#asvspoof-acquisition-and-overlap-audit) for the acquisition and audit commands.

## Baseline evaluation protocol

- Fit on the strict train manifest. Use strict development for preprocessing choices, model selection and the operating threshold. Freeze those choices before evaluating the strict test set.
- Compare a duration, level and quiet-frame baseline with a model using pooled log-mel features. Repeat with a consistent quiet-edge policy so any measured benefit can be examined independently of padding. Fit feature normalization using train data.
- Report equal error rate, ROC-AUC and errors at the development-selected threshold. Report counts and results separately for seen and unseen algorithms, using the same bona fide reference clips. Where practical, estimate uncertainty by resampling speakers. The seen-algorithm group has 15 clips, so its rates will be noisy.
- Keep the perception track on observed ratings. The 22 Behavioral Signals clips serve as an already-explored confound check.

## Reproduce the baseline

Run these commands from the repository root in the project Python environment. Choose a new output directory for each run.

```sh
python scripts/asvspoof_baseline.py train data/asvspoof2019-pilot runs/asvspoof-preflight runs/asvspoof-baseline
python scripts/asvspoof_baseline.py evaluate data/asvspoof2019-pilot runs/asvspoof-preflight runs/asvspoof-baseline
```

Each fit starts at zero and uses the same L2-regularized logistic regression. We compare duration, RMS level and quiet-frame fraction alone, mean log-mel features alone, and both sets combined. Each input is evaluated with whole clips and with quiet edges trimmed using the first and last 25 ms frames at or above −50 dBFS. Internal quiet frames remain. The script fits scaling and weights on training data and selects thresholds on development data by minimizing balanced error, favoring fewer false alarms in ties. The separate evaluation command reads the saved parameters and checks the audit snapshot and source file hashes.

## Held-out results

| Features | Whole clip ROC-AUC | Trimmed ROC-AUC |
| --- | ---: | ---: |
| Duration, level and quiet frames | 0.889 | 0.763 |
| Pooled log-mel | 0.814 | 0.608 |
| Both | 0.865 | 0.646 |

The combined whole-clip model had the highest development AUC (0.963). On evaluation, adding log-mel to simple cues changed AUC by −0.024, with a paired speaker-bootstrap 95% interval of −0.064 to +0.014. At its development-selected threshold, the combined model flagged 2 of 100 genuine clips and missed 46 of 100 spoofs. The cues model flagged 29 genuine clips and missed 17 spoofs. Whole-clip EER was 18% for cues, 26% for log-mel and 20% for both.

The cues model's AUC was 0.738 on the seen-algorithm subset and 0.915 on the unseen-algorithm subset, using the same 100 genuine clips for both comparisons. The subsets contain 15 and 85 spoofs from different algorithms, so algorithm familiarity and attack composition are mixed in this comparison. Quiet-edge trimming reduced AUC for all three models. That shows sensitivity to preprocessing in this sample. The trimming changes the duration, level and spectral summaries together, so the drop cannot be attributed to quiet frames alone.

The run generates `model.json` with training scales, coefficients, development metrics and thresholds, `results.json` with bootstrap intervals and results by attack stratum, and `evaluation_scores.csv` for local clip-level inspection.

## Transformer transfer

The final audio phase compared the four-layer Transformer with and without masked-reconstruction pretraining. Both models used the same normalization, architecture, optimizer, 15-epoch budget and three matched seeds. They trimmed quiet edges, encoded every chunk of each recording, and pooled token embeddings by frame count. All encoder weights were trainable on the 200 labeled clips. The highest development AUC selected each seed's checkpoint. The three logits were averaged, and the development set selected each ensemble's decision threshold before evaluation.

| Model | Held-out ROC-AUC | False alarms among 100 genuine clips | Missed spoofs among 100 spoof clips |
| --- | ---: | ---: | ---: |
| Simple cues, trimmed baseline | **0.763** | 28 | 33 |
| Transformer, random initialization | 0.727 | 23 | 46 |
| Transformer, pretrained initialization | 0.714 | 25 | 44 |

Pretraining changed ROC-AUC by −0.012 compared with random initialization. The paired 95% speaker-bootstrap interval spans −0.050 to +0.023. The 200 evaluation clips test within-corpus generalization on the audited ASVspoof partition. Broader performance remains unmeasured on an independent corpus. The run generates `frozen.json` and `results.json` in `runs/asvspoof-transfer/`.

### Listener ratings

Use the 44 Lavan clips from 22 speaker pairs that passed the ASVspoof train/development speaker exclusion. Keep the encoders frozen and fit regularized linear readouts. Cross five speaker folds with five listener folds, training each readout on the opposite sides of both folds. The target is the mean rating of one held-out clip from the ten held-out listeners in that fold. Each endpoint has 220 held-out clip-and-listener-group means.

| Predictor | Realness mean absolute error | Perceived trustworthiness mean absolute error |
| --- | ---: | ---: |
| Training mean | 8.86 | **6.69** |
| Known voice provenance | **8.81** | 6.83 |
| Pooled log-mel features | 9.57 | 7.01 |
| Frozen random Transformer | 9.27 | 7.09 |
| Frozen pretrained Transformer | 8.88 | 7.26 |

Errors are measured in points on the original 0–100 scale. The pretrained embeddings did not improve on the training mean in this sample. The folds reuse speakers and listeners, and the study uses a small English stimulus set. These estimates describe held-out rating means in this study. The run generates the aggregate report and fold predictions in `runs/lavan-transfer/`.

### Reproduce the final phase

Run from the repository root using the pretraining environment. Choose new output directories for a rerun.

```sh
export OPENBLAS_NUM_THREADS=1
python scripts/asvspoof_transfer.py fit data/asvspoof2019-pilot \
  runs/asvspoof-preflight runs/asvspoof-pretrain/encoder.npz runs/asvspoof-transfer
python scripts/asvspoof_transfer.py evaluate data/asvspoof2019-pilot \
  runs/asvspoof-preflight runs/asvspoof-pretrain/encoder.npz runs/asvspoof-transfer
python scripts/lavan_transfer.py data/lavan2025 \
  runs/asvspoof-preflight runs/asvspoof-pretrain/encoder.npz runs/lavan-transfer
```
