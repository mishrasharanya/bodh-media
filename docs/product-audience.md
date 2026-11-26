# Test audience responses to a product scenario

Bodh can present one of two versions of a text-based product experience, record an anonymous answer to a fixed question, and estimate the share of each audience likely to choose a defined answer under the same study conditions. Forecasts are calibrated on observed responses and checked against new participants. A scenario with no observed responses starts with `needs_calibration` in each group and version.

The [example study](../examples/bodh_review_study.json) presents two wordings of Bodh’s evidence status. Both indicate that the claim is awaiting verification. The study asks whether the participant understands that they should still review the claim, with separate groups for first-time and returning users.

## Run a study locally

Run from the repository root in the Bodh Python environment. Create a new output folder for the study. Use an anonymous study code for each participant, and use the same coding scheme across calibration and evaluation so Bodh can detect reuse. Assign the audience based on the study criteria before showing the result.

```sh
mkdir -p runs/product-review-study
python scripts/product_audience.py session examples/bodh_review_study.json \
  runs/product-review-study/calibration.csv example-code first_time
```

Replace `example-code` with a participant’s anonymous code and repeat with independent participants. For a preview, use a separate response filename or exit without saving. Bodh assigns one version by hashing the study and participant code. The session displays a single version and records the answer if the participant chooses to save it. Reports contain aggregate numbers. The local response CSV contains the anonymous codes used for the overlap check.

Freeze a forecast using the calibration wave. Run the same study with a new group of participants, recording their choices in `held_out.csv`, and then compare the two.

```sh
python scripts/product_audience.py forecast examples/bodh_review_study.json \
  runs/product-review-study/calibration.csv runs/product-review-study/forecast.json --panel-size 10
python scripts/product_audience.py evaluate examples/bodh_review_study.json \
  runs/product-review-study/calibration.csv runs/product-review-study/forecast.json \
  runs/product-review-study/held_out.csv runs/product-review-study/results.json
```

Choose new report filenames on reruns. For each audience and version, the forecast reports an expected response rate and the middle 80% of the predicted number of target responses in a panel of the chosen size, using a beta-binomial model with a uniform prior. Evaluation reports predicted and observed rates plus Brier scores, which measure the mean squared error of the probabilities. The comparison baseline pools audiences within each version. The study file is hashed so that a wording or task change requires its own calibration.

## Historical backtest

The first backtest uses the observed Lavan Experiment 1b ratings. The example outcome is whether a listener gave a realness rating of at least 60 out of 100 for a human recording or a clone. Listener groups are formed from ratings on separate screening clips. Each person contributes one sampled rating to calibration or evaluation. The two stages use disjoint listeners and speaker pairs.

```sh
python scripts/lavan_audience_backtest.py data/lavan2025/experiment1b runs/product-audience-lavan
python scripts/product_audience.py forecast runs/product-audience-lavan/study.json \
  runs/product-audience-lavan/calibration.csv runs/product-audience-lavan/forecast.json
python scripts/product_audience.py evaluate runs/product-audience-lavan/study.json \
  runs/product-audience-lavan/calibration.csv runs/product-audience-lavan/forecast.json \
  runs/product-audience-lavan/held_out.csv runs/product-audience-lavan/results.json
```

On 24 held-out responses, the audience-specific forecast scored **0.235 Brier**, compared with **0.251** for the pooled baseline. Predicted group rates differed from the observed rates by 17 to 33 percentage points, with six held-out responses per cell. This is a small retrospective listening-study backtest. The Bodh wording study has its own target outcome and needs its own participant responses.
