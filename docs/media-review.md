# Audio authenticity and claim review

Bodh can analyze an MP3 and look up evidence for supplied text in one report. The audio channel uses the frozen, quiet-edge-trimmed simple-cue classifier. Its logit and decision describe the classifier's assessment of the audio. The claim channel reads one user-supplied claim per line. It retrieves published AVeriTeC evidence and a labeled historical verdict for a unique exact match. It returns candidate evidence and a review status for unfamiliar claims. The supplied text is not automatically transcribed or aligned to the clip.

## Independent audio pilot

Bodh downloaded 200 clips from the authors' In-the-Wild archive, with 100 genuine and 100 synthetic clips across 54 speakers. The sample balances the two categories within each speaker. The public metadata contains speaker IDs and labels but no parent-recording field, so dependence between excerpts from the same parent remains possible. The audit found zero exact file or decoded-waveform matches against the 600 audited ASVspoof clips. [Publisher archive](https://huggingface.co/datasets/mueller91/In-The-Wild/tree/eee168f92c367f8c82ff2cf42b6f61e362fd6211) · [Dataset paper](https://arxiv.org/abs/2203.16263)

The frozen baseline reached **0.475 ROC-AUC** and **46% balanced accuracy**, flagging 54 of 100 genuine clips and missing 54 of 100 synthetic clips. On the trimmed ASVspoof holdout its AUC was 0.763. The change reflects this combination of dataset, sample and preprocessing. It does not isolate a causal factor.

## Evidence retrieval pilot

On the 500-claim AVeriTeC development file, 486 claims had at least one annotated evidence URL. Lexical ranking returned at least one of those URLs in the first five results for **430 of 486 claims (88.5%)**. The search pool was assembled from annotated question-answer snippets from that same split. The benchmark measures retrieval in this provided pool. It does not measure open-web search, new verdict accuracy, or transcription. The dataset and generated per-example output stay local. [AVeriTeC dataset](https://fever.ai/dataset/averitec.html)

## Run locally

Activate the project's Python environment. The public archives and benchmarks use new output paths on each rerun.

```sh
python scripts/fetch_in_the_wild.py data/in-the-wild-pilot --per-class 100 --seed 2026
python scripts/media_check.py audio-benchmark data/in-the-wild-pilot \
  runs/asvspoof-baseline/model.json runs/asvspoof-preflight runs/in-the-wild-pilot

mkdir -p data/averitec
curl -fL https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json \
  -o data/averitec/dev.json
python scripts/media_check.py evidence-benchmark data/averitec/dev.json
python scripts/media_check.py review recording.mp3 claims.txt \
  data/averitec/dev.json runs/asvspoof-baseline/model.json
```

The AVeriTeC file used in the reported run has SHA-256 `499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300`. See the [reproduction guide](reproduction.md#baselines-and-transformer) to generate the frozen baseline. The review commands accept local WAV, FLAC and MP3 clips up to 30 seconds.

## One-file review with local Whisper

`review-whisper` decodes the recording, runs whisper.cpp locally, and looks up evidence using the timestamped transcript segments. Bodh's original classifier still provides the separate authenticity score. The default is the pretrained English `base.en` model. The weights stay under the ignored `data/` directory. [whisper.cpp](https://github.com/ggml-org/whisper.cpp) · [Model file](https://huggingface.co/ggerganov/whisper.cpp/blob/0b364b566045a405be7225ee1e415a073e04da77/ggml-base.en.bin)

With `whisper-cli` on your PATH, download the model once:

```sh
mkdir -p data/whisper
curl -fL https://huggingface.co/ggerganov/whisper.cpp/resolve/0b364b566045a405be7225ee1e415a073e04da77/ggml-base.en.bin \
  -o data/whisper/ggml-base.en.bin
```

The published SHA-256 is `a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002`. After downloading the weights, the command runs offline:

```sh
python scripts/media_check.py review-whisper recording.mp3 \
  data/averitec/dev.json runs/asvspoof-baseline/model.json
```

Transcript timestamps are in seconds from the original clip. Review names, numbers and negations before interpreting evidence matches. Use `--whisper-model` with a multilingual GGML model and `--language` to change the transcription language.

## Optional Google transcription

`review-google` sends the clip's mono audio to Google Cloud Speech-to-Text v1. Bodh scores authenticity locally and uses the returned transcript segments to look up AVeriTeC references or retrieve candidate evidence. Review the transcript before relying on it, especially dates, numbers and negations. The standard v1 API lists the first 60 minutes per month per billing account as free, including the tier without data logging. Billing must be enabled and usage beyond the free allowance can incur charges. [Pricing](https://cloud.google.com/speech-to-text/pricing) · [Authentication](https://cloud.google.com/speech-to-text/docs/v1/authentication)

With the Google Cloud CLI installed and a billing-enabled project you control, run:

```sh
gcloud services enable speech.googleapis.com --project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID

python scripts/media_check.py review-google recording.mp3 \
  data/averitec/dev.json runs/asvspoof-baseline/model.json --project YOUR_PROJECT_ID
```

Use `--language` to change the speech language from `en-US`. The original `review` command uses your supplied text and works offline.
