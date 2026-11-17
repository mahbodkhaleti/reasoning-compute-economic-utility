# Human annotation: notes for the authors

This folder is for you only. **Do not send it to annotators.** `answer_key.json` contains the model names, the parser and reader output, the LLM labels and the true expected profits.

## What to send

Two items:

- `annotator_package.zip`, which contains:
  - `INSTRUCTIONS.md`
  - `traces.html`
  - `annotation_workbook.xlsx`
  - the CSV fallbacks
- the text in `MESSAGE_TO_ANNOTATORS.txt`, as your email body.

Two annotators, each labelling all 50 traces, give you human-to-human agreement as well as human-to-machine agreement. With only one annotator, the score still reports agreement with the parser, the text reader and the LLM labels.

Ask for the files back by 14 November. That leaves two days to score them and update the paper before the 16 November deadline.

## The 50 traces

- Drawn with a fixed seed from the 150 traces the LLM annotators labelled, stratified by model (9, 9, 8, 8, 8, 8) and by budget (512, 768 and unforced).
- Script: `analysis/revision/t15_make_human_package.py`.
- Trace IDs are shuffled, and nothing in the annotator files names the model or the budget.
- Rows: 310 options in total.

## Scoring

```
python3 score_human_labels.py answer_key.json annotation_workbook_AB.xlsx annotation_workbook_CD.xlsx
```

It writes `human_agreement.json` and reports:

- **Option level:**
  - "written" agreement and Cohen's κ against the 2% parser, the text reader and the LLM annotators;
  - how often the reader's value matches the human's;
  - the human-measured miscompute rate.
- **Trace level:**
  - finished-list agreement with the parser;
  - how often the human "trace_best" matches the hybrid written-best used in Table 2.
- **Between annotators:** κ for "written", plus agreement on trace_best and finished.

I tested the scorer on two simulated workbooks.

## For the paper

Once the scores are in, replace "The annotators are LLMs" and the "LLM labels are a check, not ground truth" sentence with one sentence of this form:

"Two human annotators labelled 50 of these traces (310 options); against them the parser has precision X and recall Y (κ = Z), the text reader reads W% of values correctly, and the written-best option used in Table 2 matches theirs in V% of traces (human–human κ = U)."

Fill in X, Y, Z, W, V and U from `human_agreement.json`.
