"""Report audio provenance and separately look up evidence for supplied claims."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bodh import audio, baseline, google_transcription, whisper_transcription
from bodh.claim_evidence import EvidenceIndex, read_records
from bodh.eda import sha256
from bodh.manifest import read_assets, write_csv


def model_at(path: Path) -> dict:
    model = json.loads(path.read_text())["policies"]["trimmed"]["cues"]
    if tuple(model["features"]) != baseline.CUES:
        raise ValueError("Expected Bodh's frozen quiet-edge-trimmed cue baseline")
    return model


def score(path: Path, model: dict) -> dict:
    return score_decoded(audio.load_audio(path), model)


def score_decoded(decoded: audio.Audio, model: dict) -> dict:
    summary, _ = audio.features(audio.trim_quiet_edges(decoded))
    logits = baseline.scores(model, np.array([[summary[key] for key in baseline.CUES]]))
    threshold = model["decision_threshold"]
    return {"spoof_logit": float(logits[0]), "decision_threshold": threshold,
            "flagged_as_synthetic": bool(logits[0] >= threshold),
            "duration_s": decoded.duration}


def external(root: Path, model_path: Path, audit: Path, out: Path) -> None:
    import csv
    from hashlib import sha256 as digest

    inventory = {"sha256": set(), "pcm_16k_mono_sha256": set()}
    for path in sorted(audit.glob("*_integrity.csv")):
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                for key, values in inventory.items():
                    values.add(row[key])
    if not all(inventory.values()):
        raise ValueError("No audited ASVspoof fingerprints found")
    source = root / "manifest.csv"
    assets = read_assets(source)
    model = model_at(model_path)

    def inspect(asset) -> dict:
        path = root / asset.path
        raw = sha256(path)
        clip = audio.load_audio(path)
        pcm = digest(clip.mono.astype("<f4").tobytes()).hexdigest()
        if raw in inventory["sha256"] or pcm in inventory["pcm_16k_mono_sha256"]:
            raise ValueError(f"External recording matches audited ASVspoof audio: {asset.asset_id}")
        summary, _ = audio.features(audio.trim_quiet_edges(clip))
        logits = baseline.scores(model, np.array([[summary[key] for key in baseline.CUES]]))
        return {"asset_id": asset.asset_id, "label": asset.condition, "speaker": asset.speaker_id,
                "spoof_logit": float(logits[0]), "sha256": raw, "pcm_16k_mono_sha256": pcm}

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(inspect, assets))
    if any(len({row[key] for row in rows}) != len(rows) for key in inventory):
        raise ValueError("Repeated audio within the external sample needs review")
    labels = np.array([int(row["label"] == "spoof") for row in rows])
    logits = np.array([row["spoof_logit"] for row in rows])
    report = {
        "model": "quiet-edge-trimmed simple-cue logistic regression",
        "model_sha256": sha256(model_path), "manifest_sha256": sha256(source),
        "threshold": model["decision_threshold"], "sampled_speakers": len({r["speaker"] for r in rows}),
        "exact_file_or_decoded_pcm_overlap_with_asvspoof": 0,
        **baseline.metrics(labels, logits, model["decision_threshold"]),
    }
    out.mkdir(parents=True)
    write_csv(out / "scores.csv", tuple(rows[0]), rows)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    combined = commands.add_parser("review")
    for key in ("audio", "claims", "averitec", "model"):
        combined.add_argument(key, type=Path)
    google = commands.add_parser("review-google", help="Send clip audio to Google for transcription")
    for key in ("audio", "averitec", "model"):
        google.add_argument(key, type=Path)
    google.add_argument("--project", required=True, help="Google Cloud billing/quota project ID")
    google.add_argument("--language", default="en-US", help="BCP-47 speech language")
    whisper = commands.add_parser("review-whisper", help="Transcribe the clip locally with whisper.cpp")
    for key in ("audio", "averitec", "model"):
        whisper.add_argument(key, type=Path)
    whisper.add_argument("--whisper-model", type=Path, default=Path("data/whisper/ggml-base.en.bin"))
    whisper.add_argument("--language", default="en", help="Whisper language code")
    benchmark = commands.add_parser("evidence-benchmark")
    benchmark.add_argument("averitec", type=Path)
    evaluator = commands.add_parser("audio-benchmark")
    for key in ("root", "model", "audit", "out"):
        evaluator.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.command == "audio-benchmark":
        external(args.root, args.model, args.audit, args.out)
    else:
        index = EvidenceIndex(read_records(args.averitec))
        if args.command == "review":
            claims = [line.strip() for line in args.claims.read_text().splitlines() if line.strip()]
            if not claims:
                parser.error("Supply at least one transcribed claim, one per line")
            result = {"audio_authenticity": score(args.audio, model_at(args.model)),
                      "claim_evidence": [index.report(claim) for claim in claims],
                      "claim_input": "User-supplied text; relationship to the recording is unverified"}
        elif args.command in {"review-google", "review-whisper"}:
            clip = audio.load_audio(args.audio)
            authenticity = score_decoded(clip, model_at(args.model))
            try:
                if args.command == "review-whisper":
                    provider = "whisper.cpp (local)"
                    segments = whisper_transcription.recognize(clip, args.whisper_model, args.language)
                else:
                    provider = "Google Cloud Speech-to-Text v1"
                    segments = google_transcription.recognize(clip, args.project, args.language)
            except (OSError, ValueError) as error:
                parser.exit(2, f"{args.command}: {error}\n")
            result = {
                "audio_authenticity": authenticity,
                "transcription": {"provider": provider, "segments": segments},
                "claim_evidence": [
                    {**index.report(segment["text"]), "start": segment["start"], "end": segment["end"]}
                    for segment in segments
                ],
                "claim_input": "Transcribed speech segments; review wording and timestamps",
            }
        else:
            result = index.benchmark()
        print(json.dumps(result, indent=2, ensure_ascii=False))
