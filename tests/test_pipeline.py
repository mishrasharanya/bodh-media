"""Exercise the public workflows with generated audio and synthetic ratings."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

from bodh import baseline
from bodh.audio import features, load_audio, trim_quiet_edges
from bodh.claim_evidence import EvidenceIndex
from bodh.manifest import COLUMNS, write_csv
from bodh.perception import TRIAL_COLUMNS


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest = self.root / "manifest.csv"
        write_csv(self.manifest, COLUMNS, [
            dict(zip(COLUMNS, ("fixture", "tone", "audio", "tone.wav",
                               "unsplit", "fixture", "reference", "", ""))),
        ])

    def cli(self, *args):
        return subprocess.run([sys.executable, "-m", "bodh.cli", *map(str, args)],
                              capture_output=True, text=True, check=True)

    def test_cli_extracts_audio_and_writes_a_report(self):
        rate = 22050
        tone = 12000 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
        path = self.root / "tone.wav"
        write(path, rate, np.rint(np.r_[np.zeros(rate // 2), tone, np.zeros(rate // 2)]).astype("int16"))
        out = self.root / "report"
        self.cli("eda", self.manifest, self.root, out)
        report = json.loads((out / "report.json").read_text())
        self.assertEqual((report["n_processed"], report["n_failed"]), (1, 0))
        self.assertTrue((out / "distributions.png").is_file())
        clip = load_audio(path)
        trimmed = trim_quiet_edges(clip)
        self.assertEqual(clip.source_rate, rate)
        self.assertLess(trimmed.duration, 1.1)
        self.assertAlmostEqual(features(trimmed)[0]["active_spectral_centroid_median_hz"], 440, delta=2)

    def test_cli_joins_ratings_to_asset(self):
        trials = self.root / "trials.csv"
        write_csv(trials, TRIAL_COLUMNS, [
            dict(zip(TRIAL_COLUMNS, ("fixture", "tone", "example", "listener_1", "realness", 20))),
            dict(zip(TRIAL_COLUMNS, ("fixture", "tone", "example", "listener_2", "realness", 40))),
        ])
        report = json.loads(self.cli("perception", self.manifest, trials).stdout)
        self.assertEqual(report["groups"][0]["mean_of_stimulus_means"], 30)
        self.assertEqual(report["groups"][0]["n_trials"], 2)

    def test_equal_scores_keep_both_classes_tied(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.zeros(4)
        threshold = baseline.threshold_on_dev(labels, scores)
        result = baseline.metrics(labels, scores, threshold)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["eer"], 0.5)
        self.assertEqual(result["false_positives"], 0)

    def test_new_claim_does_not_inherit_reference_verdict(self):
        reference = {
            "claim": "The station opened in 2020.", "label": "Supported",
            "questions": [{"question": "When did the station open?", "answers": [
                {"source_url": "https://example.org/records", "answer": "The station opened in 2020."}
            ]}],
        }
        index = EvidenceIndex([reference])
        self.assertEqual(index.report("The station opened in 2020")["status"], "published_reference")
        unknown = index.report("The station opened in 2021.")
        self.assertEqual(unknown["status"], "needs_review")
        self.assertIsNone(unknown["verdict"])
        self.assertEqual(unknown["candidate_evidence"][0]["url"], "https://example.org/records")
