# Hand-labelling 100 traces (about 2–3 hours)

The sample is 100 reasoning traces from the 768-token runs, drawn at random with 15–18 per model. Each file in `traces/` shows the private value, the options and the trace, which is cut off at 768 tokens. Fill in `label_sheet.csv`, one row per trace. Don't use a calculator or any other tool to check the arithmetic in the trace; label only what the trace says.

For each option A–G that exists (`n_options` gives how many):

- **`value_written_X`:** the expected profit the trace writes for option X, as a number, for example `16.95`.
  - If the trace writes several values for X, use the last one.
  - If it writes none, leave the cell empty.
- **`value_correct_X`:** whether that written value equals (v − bid) × win probability to within 2% (or $0.05). Here you *may* compute the true value yourself to compare. Answer `Y` or `N`; leave it empty if nothing was written.

For the trace as a whole:

- **`list_finished`:** `Y` if the trace writes a value for every option, `N` otherwise.
- **`stated_choice`:** the option letter the trace itself names as its choice or as the best option, for example "the highest expected profit is Option C". If it names one more than once, use the last time. Leave it empty if the trace never names one.
- **`notes`:** optional.

Leave the other columns alone. You don't need to label the gap category: `score_hand_labels.py` derives it from your labels with the paper's rules and compares it with the code's category.

When you're done, run:

```
python3 score_hand_labels.py label_sheet.csv
```

The script reports:

- precision and recall of the code reader and of the 2% matcher against your written values;
- agreement on whether the list is finished;
- agreement on the stated choice;
- agreement on the gap category.

The paper can then say, in one sentence, that one author labelled 100 traces, and the labels are released with the code.
