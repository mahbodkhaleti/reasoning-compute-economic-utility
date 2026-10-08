# Saved-decision audit and human-study preparation

8 October 2026. This stage adds no model inference and collects no human responses.

## Completed

- Audited all 23,328 greedy fixed-budget decisions against a calculator and the prompt objective.
- Reconstructed four-decimal displayed probabilities, preserved the published calculator, and checked a calculator using exactly the displayed information.
- Added paired family uncertainty and simultaneous six-model bands for dollar-profit gaps.
- Produced product-level endpoint summaries and regret-concentration diagnostics.
- Integrated the findings into the editable manuscript and visually checked its PDF.
- Prepared a 19-profile, 12-choice offline human pilot, participant protocol, randomized presentation, response validation and exploratory analysis.
- Checked the survey code/state flow using a mock document. Live browser rendering was blocked by local-file URL policy and is not claimed.
- Prepared venue-independent submission and claim-boundary checks. Target-specific formatting remains pending.

## Findings

At 768 tokens, model expected bidder profit is $15.09–$36.00 below the calculator on average. All six simultaneous exploratory bands remain negative, including a separately recomputed comparison against the display-matched calculator. Additional tokens improve the LLM pipeline without establishing superiority over numerical optimization.

Displayed-objective optimality rises from 18.1–26.1% at zero to 42.1–86.3% at 768. Some actions optimal under the displayed distribution are suboptimal under held-out scoring. The categories describe observable objective agreement, not internal arithmetic failure. Dollar profit is expected profit in constructed tasks, not observed income.

## Reproduce this stage

With NumPy, Pandas and SciPy installed, run:

```sh
python3 analyze_saved_decisions.py
python3 export_audit.py
python3 validate_audit.py
```

The complete parent package's `reproduce_cpu_revision.sh` runs this stage after all previous CPU work. It accepts the human script's pending-data exit code 3 without marking the human-evidence gate complete.

`data/saved_greedy_decisions.csv` is a compact extraction of preserved records, with original input hashes in `data/extraction_provenance.json`. `data/scenarios.csv` and the archived analytic rows make the numerical audit self-contained. `data/prompt_formatter_reference.py` is reference material only; the audit does not import its inference dependencies.

The manuscript requires a LaTeX installation:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Human study

Open `human_study/survey.html` locally in a browser. Read `human_study/protocol.md` before recruiting or collecting responses. The researcher-only source identity key should remain separate from participant materials. A code beginning TEST marks a diagnostic session. Give genuine exported files to the researcher through an agreed channel and place them in `human_study/responses/`.

The current human-study status is **awaiting responses**. Hypothetical profile-choice data cannot by itself establish actual willingness to pay or deployed-market parameters. The 24-participant pilot target is an operational usability target, not a confirmatory power calculation.

Profile regeneration requires the original preserved stochastic records in the complete parent package. The already prepared survey, profiles and pilot analysis can be used without those raw records or any GPU.

## Remaining work

Select a target venue and author/anonymity details, conduct the human pilot with actual participants, then decide on a formal study with an appropriate population and incentives. Richer economic tasks and matched budgets above the tested ceiling still require new inference. The public repository has not been updated by this local artifact preparation.
