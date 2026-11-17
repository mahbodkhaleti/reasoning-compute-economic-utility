# Annotation instructions

Thank you for helping. This takes about 3–4 hours in total, and you can split it across several sittings.

## What this is

A language model was given an auction decision to solve. It saw three things:

- a private value (what the item is worth to it);
- 5 or 7 candidate bids, labelled A, B, C and so on;
- for each bid, an estimated probability of winning.

Its task was to pick the bid with the highest expected profit:

expected profit = (private value − bid) × win probability

Before answering, the model wrote out its reasoning. This reasoning is the "trace". There are 50 traces, numbered T01 to T50.

**Your job is to record what each trace wrote, not whether it was right.** Do not check the model's arithmetic. Do not work out the correct answer yourself.

## Files

| File | What it is |
|---|---|
| `traces.html` | The 50 traces. Open it in any web browser to read them. |
| `annotation_workbook.xlsx` | Where you record your answers. Fill in only the yellow cells. |
| `option_labels.csv`, `trace_labels.csv` | The same two answer tables, for use if you cannot open Excel files. |

The workbook has four sheets:

- **Read me**: a short version of these instructions, plus a progress counter.
- **Traces**: the same text as `traces.html`.
- **Option labels**: one row per option of every trace (310 rows).
- **Trace labels**: one row per trace (50 rows).

Row 2 of each labels sheet is a grey **example row** for an imaginary trace T00. Leave it as it is. Do not add, delete or sort any rows.

## Step by step, for each trace

1. Open the trace in `traces.html`. Note the private value and the options table at the top.
2. Read the trace text. For each option, look for the number the model gives as that option's expected profit.
3. Fill in that trace's rows in **Option labels**.
4. Fill in that trace's row in **Trace labels**.

### Option labels (one row per option)

**written (Y/N)**

- **Y** if the trace writes a number that it presents as this option's expected profit. This means the result of the multiplication. The number counts even if it is wrong, rounded or approximate, for example "about $16".
- **N** if the trace never reaches a result for this option. For example:
  - it writes only the formula;
  - it writes only the subtraction (private value − bid);
  - it is cut off before the result;
  - it never mentions the option.

**written_value**

- Copy the number exactly as the trace wrote it, without "$" or commas. Example: 1053.04.
- If the trace gives several values for the same option (for example, it recomputes it), use the **last** one.
- Leave the cell blank when written = N.

**notes**: optional. Use it for anything unclear.

### Trace labels (one row per trace)

**list_finished (Y/N)**

- **Y** if every option in that trace has written = Y.
- **N** otherwise.

**trace_best**

- The letter of the option whose **written** value is largest. Judge this only from the numbers the trace wrote; do not recompute anything.
- Write **NONE** if no option has a written value.
- For a tie, write both letters, for example **A,C**.

**stated_choice**

- If the trace itself says which option or bid it picks, write that option's letter. Examples: "So I'll go with option C", or "the best bid is $250", where $250 is option C.
- Write **NONE** if the trace never says. Many traces are cut off before they decide.
- If the trace changes its mind, use the last choice it states.

**minutes**: optional. Roughly how long the trace took you.

**notes**: optional.

## A short example

Suppose a trace with private value $2,510 contains these lines:

> Option A: 2510 − 870 = 1640. 1640 × 0.6421 = 1053.04
> Option B: 2510 − 830 = 1680. 1680 × 0.6105 ≈ 1025.6
> Option C: 2510 − 2475 = 35. 35 × 0.9368 = 32.79
> Option D: 2510 − 1890 = 620. 620 × 0.8947 = 554.7
> Option E: bid $1235. Win probability 0.7263. (

and then stops. You would record:

- A: Y, 1053.04
- B: Y, 1025.6
- C: Y, 32.79
- D: Y, 554.7
- E: N, value left blank (the result was never written)

For the trace row:

- list_finished = N
- trace_best = A
- stated_choice = NONE

## Rules

- **Do not use a calculator** to check the model's arithmetic. Record what is written.
- **Do not use ChatGPT, Claude or any other AI tool** at any point. The aim is an independent human check.
- **Work alone.** If someone else is annotating the same traces, do not compare answers until you have both finished.
- Do not look up anything about the models or the study.
- If something is ambiguous, make your best call and explain it in notes.

## When you are done

1. Check the counter on the **Read me** sheet. It should show 310 of 310 option rows and 50 of 50 trace rows.
2. Save the file as `annotation_workbook_<your initials>.xlsx`.
3. Send it back to the person who gave you this package.
