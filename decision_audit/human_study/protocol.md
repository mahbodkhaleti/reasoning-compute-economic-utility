# Human service-choice pilot: prepared, not conducted

8 October 2026. This protocol uses existing model outputs and requires no new inference. No participants have been recruited and no human responses have been collected.

## Question and claim boundary

Do people respond to mean utility-ratio performance, dollar-profit performance and hypothetical fees when choosing between anonymous decision services? Can a relative-quality selection rule or a quality-difference rule describe their responses to the displayed benchmark information?

The pilot measures choices among constructed, explicitly hypothetical benchmark profiles. It does not identify deployed service selection, actual willingness to pay, real income, auction total surplus or a calibrated market equilibrium. Model outputs are stimuli, not human ratings. Pilot estimates must not be inserted into the manuscript as confirmatory market parameters.

## Materials

- `survey.html`: standalone offline participant interface, with no network requests or automatic response submission.
- `profiles_key.json`: researcher-only mapping of 18 model/budget profiles and a calculator to source identities. Do not distribute this key to participants before they complete the survey.
- `trial_design.json`: anonymized displayed statistics and fixed pilot comparisons.
- `build_provenance.json`: hashes and protocol for the preserved generations used to construct profiles.
- `analyze_human_choices.py`: validates exports, excludes diagnostic sessions and produces exploratory participant-clustered analyses.

The model profiles use the stochastic frontier at 128/384/768 tokens, averaged over 216 scenarios and five recorded generation seeds. The calculator uses the construction-split policy scored on the same held-out tasks. Displayed utility and accuracy are rounded to 0.1 percentage points; displayed mean dollar profit is rounded to cents. Analysis uses these displayed values, not hidden extra precision. Utility ratio is labelled as the fraction of feasible expected profit captured, never as win probability or reliability.

## Pilot design

An operational pilot target of 24 adult participants is proposed to check comprehension, interface behavior and whether the contrasting information produces varied choices. This target is not a power calculation. The participant population and recruitment channel must be selected before collection; a convenience pilot cannot establish population-level client behavior.

Each participant completes 12 comparisons. Six use equal hypothetical service fees and are primary for exploratory quality-selection fitting. They deliberately include utility-versus-dollar ranking conflicts, small and large quality differences, and a calculator comparison. Six use unequal fees and examine hypothetical price tradeoffs. Equal-fee trials occur first; trial order within each block and left/right service presentation are randomized deterministically by anonymous participant code. This block order is fixed, so a causal comparison of blocks is not claimed.

Pairs are contrast-selected rather than sampled from a market distribution. Both forced-choice options remain available; there is no outside option. Results therefore concern conditional A-versus-B preference, not purchase incidence. No payment or real service delivery occurs in the current interface. Hypothetical price responses must not be called incentive-compatible willingness to pay.

Participants read the task definitions and voluntarily agree before starting. The interface collects only an anonymous code, choices, presentation details and response durations. Export happens only when the participant clicks the export button. The researcher should give participants an agreed return channel and explain any separately arranged compensation before collection.

## Analysis prepared before data

Fit separate competing single-score models: logistic choice based on the log utility-ratio difference (Tullock form), on the utility-ratio difference, and on the normalized dollar-profit difference. Include a left-side preference term. The fee block additionally includes a fee-difference term. Use participant-level bootstrap draws, retaining each participant's complete response vector. These models are compared descriptively in the pilot; no winner is selected as a confirmed market mechanism.

Nonnegative quality slopes and nonpositive fee slopes are assumed in the prepared fits. Coefficients are bounded to avoid presenting a separated likelihood as a precise finite estimate; boundary estimates are flagged. These assumptions and bounds need to be retained or changed explicitly before a formal study. Timing alone is not used to exclude a participant. Duplicate codes, incomplete exports, unknown profiles and inconsistent fee/choice fields fail validation rather than being silently repaired.

Codes beginning `TEST` and exports marked as diagnostic never count as observed participants. The analysis returns a pending status when no genuine responses exist. A simulated or diagnostic interface test cannot pass the human-evidence gate.

## Run and collect

Open `survey.html` in a browser. Assign an anonymous code, allow the participant to complete the survey, and collect their exported `choice_<code>.json` through the agreed channel. Store genuine exports in `responses/`.

With NumPy and SciPy installed:

```sh
python3 analyze_human_choices.py --responses responses --draws 500
```

Exit code 3 means no observed human responses; this is a pending-data status. Exit code 0 means the exploratory pilot analysis completed, not that the scientific gate passed.

## Gate for a stronger study

After the pilot, inspect comprehension and separation, check whether the chosen profiles distinguish the competing mappings, and define the target participant/client population. Before a formal study, freeze recruitment, incentives or explicit hypothetical framing, exclusions, primary outcome, held-out prediction procedure, and a sample-size calculation for the intended claim. To calibrate actual service demand or client value, add representative users and real choice/payment evidence. The current pilot cannot supply those facts by itself.
