"""Run a local audience test and check a frozen response forecast against fresh observations."""

import argparse
import csv
import fcntl
import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np
from scipy.stats import betabinom

from bodh.eda import sha256

COLUMNS = ("study_sha256", "participant_id", "audience", "variant", "response")


@dataclass(frozen=True)
class Observation:
    participant_id: str
    audience: str
    variant: str
    response: int


def load_study(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in ("name", "task", "outcome")):
        raise ValueError("Specify the study name, task, and outcome")
    for key, minimum in (("audiences", 1), ("variants", 2)):
        entries = spec.get(key)
        if (not isinstance(entries, dict) or len(entries) < minimum
                or any(not name.strip() or not isinstance(text, str) or not text.strip()
                       for name, text in entries.items())):
            raise ValueError(f"Specify at least {minimum} named {key} with nonempty descriptions")
    choices = spec.get("choices")
    if (not isinstance(choices, list) or len(choices) != 2
            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
            or choices[0] == choices[1] or spec.get("target_choice") not in choices):
        raise ValueError("Specify two distinct choices and which one counts toward the outcome")
    return spec


def read_responses(path: Path, spec: dict, fingerprint: str) -> list[Observation]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError(f"Response CSV requires {', '.join(COLUMNS)}")
        seen = set()
        observations = []
        for row in reader:
            participant = row["participant_id"]
            if (row["study_sha256"] != fingerprint or not participant or not participant.strip()
                    or participant in seen or row["audience"] not in spec["audiences"]
                    or row["variant"] not in spec["variants"] or row["response"] not in {"0", "1"}):
                raise ValueError(f"Invalid or duplicate response at {path.name} line {reader.line_num}")
            seen.add(participant)
            observations.append(Observation(participant, row["audience"], row["variant"], int(row["response"])))
        return observations


def session(study: Path, responses: Path, participant: str, audience: str) -> None:
    spec, fingerprint = load_study(study), sha256(study)
    if audience not in spec["audiences"] or not participant.strip():
        raise ValueError("Provide a defined audience and a nonempty anonymous participant code")
    allocation = digest(f"{fingerprint}:{participant}".encode()).digest()
    variant = list(spec["variants"])[int.from_bytes(allocation[:8], "big") % len(spec["variants"])]
    print(f"{spec['name']}\n\n{spec['variants'][variant]}\n\n{spec['task']}")
    for number, choice in enumerate(spec["choices"], 1):
        print(f"{number}. {choice}")
    answer = input("Choose 1 or 2 to save an anonymous response, or press Enter to exit: ").strip()
    if not answer:
        return
    if answer not in {"1", "2"}:
        raise ValueError("Choose 1 or 2")
    participant_response = int(spec["choices"][int(answer) - 1] == spec["target_choice"])
    with responses.open("a+", newline="", encoding="utf-8") as destination:
        fcntl.flock(destination, fcntl.LOCK_EX)
        destination.seek(0, 2)
        writer = csv.writer(destination)
        if not destination.tell():
            writer.writerow(COLUMNS)
        elif any(row.participant_id == participant for row in read_responses(responses, spec, fingerprint)):
            raise ValueError("This participant already responded in this response file")
        writer.writerow((fingerprint, participant, audience, variant, participant_response))


def save(path: Path, report: dict) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")


def forecast(study: Path, calibration: Path, out: Path, panel_size: int) -> None:
    if panel_size < 1:
        raise ValueError("Panel size must be positive")
    spec, fingerprint = load_study(study), sha256(study)
    data = read_responses(calibration, spec, fingerprint)
    cells = defaultdict(list)
    pooled = defaultdict(list)
    for row in data:
        cells[row.audience, row.variant].append(row.response)
        pooled[row.variant].append(row.response)
    predictions = {}
    for audience in spec["audiences"]:
        predictions[audience] = {}
        for variant in spec["variants"]:
            observed = cells[audience, variant]
            n, successes = len(observed), sum(observed)
            if n == 0:
                predictions[audience][variant] = {"status": "needs_calibration", "responses": 0}
                continue
            a, b = successes + 1, n - successes + 1
            predictions[audience][variant] = {
                "status": "estimated", "responses": n, "observed_target_choices": successes,
                "predicted_rate": a / (a + b),
                "predicted_panel_count_middle_80_percent": betabinom.ppf([0.1, 0.9], panel_size, a, b).astype(int).tolist(),
                "pooled_variant_rate": (sum(pooled[variant]) + 1) / (len(pooled[variant]) + 2),
            }
    save(out, {"study_sha256": fingerprint, "study": spec["name"], "outcome": spec["outcome"],
               "calibration_sha256": sha256(calibration), "panel_size": panel_size,
               "prior": "Beta(1, 1)", "predictions": predictions})


def evaluate(study: Path, calibration: Path, frozen: Path, held_out: Path, out: Path) -> None:
    spec, fingerprint = load_study(study), sha256(study)
    prediction = json.loads(frozen.read_text(encoding="utf-8"))
    if prediction["study_sha256"] != fingerprint or prediction["calibration_sha256"] != sha256(calibration):
        raise ValueError("Study or calibration responses differ from the frozen forecast")
    calibration_ids = {row.participant_id for row in read_responses(calibration, spec, fingerprint)}
    observed = read_responses(held_out, spec, fingerprint)
    if not observed:
        raise ValueError("Evaluation requires real held-out responses")
    if calibration_ids.intersection(row.participant_id for row in observed):
        raise ValueError("Held-out responses reuse participants from calibration")
    errors, pooled_errors = [], []
    cells = defaultdict(list)
    for row in observed:
        cells[row.audience, row.variant].append(row.response)
    comparison = {}
    for (audience, variant), values in sorted(cells.items()):
        cell = prediction["predictions"][audience][variant]
        result = {"responses": len(values), "observed_target_choices": sum(values),
                  "observed_rate": sum(values) / len(values), "forecast_status": cell["status"]}
        if cell["status"] == "estimated":
            rate, pooled = cell["predicted_rate"], cell["pooled_variant_rate"]
            result.update(predicted_rate=rate, rate_error=result["observed_rate"] - rate)
            errors.extend((value - rate) ** 2 for value in values)
            pooled_errors.extend((value - pooled) ** 2 for value in values)
        comparison.setdefault(audience, {})[variant] = result
    save(out, {"study_sha256": fingerprint, "forecast_sha256": sha256(frozen),
               "held_out_sha256": sha256(held_out), "held_out_participants": len(observed),
               "scored_participants": len(errors),
               "audience_brier": float(np.mean(errors)) if errors else None,
               "pooled_variant_brier": float(np.mean(pooled_errors)) if pooled_errors else None,
               "comparison": comparison})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    collect = phases.add_parser("session")
    collect.add_argument("study", type=Path)
    collect.add_argument("responses", type=Path)
    collect.add_argument("participant", help="Anonymous study code")
    collect.add_argument("audience")
    fit = phases.add_parser("forecast")
    for argument in ("study", "calibration", "out"):
        fit.add_argument(argument, type=Path)
    fit.add_argument("--panel-size", type=int, default=10)
    test = phases.add_parser("evaluate")
    for argument in ("study", "calibration", "forecast", "held_out", "out"):
        test.add_argument(argument, type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "session":
            session(args.study, args.responses, args.participant, args.audience)
        elif args.phase == "forecast":
            forecast(args.study, args.calibration, args.out, args.panel_size)
        else:
            evaluate(args.study, args.calibration, args.forecast, args.held_out, args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
