# Map of the eBay pipeline (what AuctionNet must replicate)

Sources: `pipeline/orig/common.py`, which is `robustness_extensions/scripts/common.py`, used for Phi-mini, Qwen3-4B, Qwen3-8B and DeepSeek-8B. Granite-8B and Qwen3-14B use `pipeline/new/common.py`, which is `new_models_full_server_package/src/common.py`. Scenarios come from `pipeline/orig/prepare_scenarios.py`. All three files are copied unchanged.

## Scenarios (`prepare_scenarios.py`)

- **Clean auctions.** Each category is reduced to one row per auction with `final_price` and `n_bidders`. Auctions with an inconsistent opening bid are dropped.
- **Split.** A random 70/30 split within each category, with seed `stable_int(20260819, category, "split")`.
- **Values.** `value_grid(train_prices, 18, step)`: from `ceil(q35/step)*step` to `ceil(1.25*q95/step)*step`, with 18 values linearly spaced, rounded to the step and deduplicated.
- **Bid step.** A single global `bid_step = 5`, not per product.
- **Candidate menus.** `independent_candidates(value, train_prices, K, step, seed)` builds a grid from `floor(q02/step)*step` to `min(value - step, ceil(q98/step)*step)`.
  - If `seed % 2 == 0` and the grid is long enough, the menu is a random contiguous window.
  - Otherwise it is a random sample.
  - **Discrepancy with the task spec:** window or sample is decided by seed parity. It is *not* "one window and one sample per value and K". We follow the code.
  - Menus never condition on the optimum.
- **Menu seed.** `stable_int(20260819, category, value, K, set_index, "candidates")`, with K in {5, 7} and set_index in {0, 1}.
- **Win probabilities.** `train_p = mean(train_prices <= b)` and `test_p = mean(test_prices <= b)`, from `empirical_probability`.
- **Scenario id.** `stable_int(category, value, K, set_index, modulo=1e12)`, written as 12 digits.
- **Output schema.** `scenarios.csv` has the columns `scenario_id, category, value, option_count, candidate_set_index, candidate_seed, train_best_sorted_position, test_best_sorted_position, candidates_json`.

## Decisions (`run_fixed_budget.py` + `ModelHarness`)

- **Option orders.** `shuffled_candidates(scenario, repeat_seed)` uses `random.Random(stable_int(scenario_id, repeat_seed, "", "option-order"))`. The repeat seeds are 11, 23 and 37, giving 648 decisions per model and budget.
- **Prompt (`auction_prompt`).** It has *no item description*:
  ```
  You are a bidder in a first-price auction.
  Your private value is ${value:.2f}.

  Available bids are shown in randomized order:
  Option A: bid ${bid:.2f}; estimated win probability {train_p:.4f}
  ...

  For any bid b, expected profit under the supplied estimate is
  (private value - b) * estimated win probability.
  Choose one listed bid. Do not infer the answer from option position.
  End exactly with: FINAL BID: <number>
  ```
  **Discrepancy with the task spec:** the spec asks to "change only the item description", but there is none. AuctionNet therefore uses this template verbatim, `$` sign included. Values and bids are given per 1,000 impressions (CPM).
- **Chat template.** `apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, <thinking flag>)`.
  - orig package: `enable_thinking=True`, which templates that don't use it ignore.
  - new package: Qwen3-14B passes `enable_thinking=True` and Granite passes `thinking=True`.
  - Prompt ids are `tokenizer(prefix)["input_ids"]`, with default special tokens.
- **Forced budget c > 0.** `generate(min_new_tokens=c, max_new_tokens=c, bad_words_ids=[[tokenizer.eos_token_id]], greedy)`.
  - `min_new_tokens` also blocks the generation-config end-of-sequence ids until c tokens are reached.
  - **The end-of-thinking tag (`</think>`) is NOT suppressed.** A forced trace can close its thinking block and carry on.
- **Budget 0.** No reasoning tokens.
- **Free-text answer.** The suffix `"\n\nNow give the required answer. FINAL BID:"` is appended to the generated ids (not re-tokenized), followed by 20 greedy tokens. `parse_final_bid` reads the bid.
- **Unforced run.** One `generate(max_new_tokens=cap)` with normal stopping. `reasoning_text` is the whole decoded output (`skip_special_tokens=True`).
- **Letter readout (`constrained_bid_readout`), which is the paper's answer.**
  - Input: `tokenizer(chat prefix)` + `tokenizer(reasoning_text, add_special_tokens=False)`, so the reasoning is decoded and re-tokenized, + `tokenizer("\n\nUse the reasoning above and select the best displayed option. Return one option letter only. FINAL OPTION:", add_special_tokens=False)`.
  - One forward pass; softmax over the label tokens.
  - Each label's token is the first single-token variant among `" X"` and `"X"`.
  - No end-of-thinking tag is inserted.
- **Scoring.** `score_bid`; `test_utility_ratio` is q.
- **Decoding and hardware.** Greedy, batch size 1, bf16, HF transformers. Runs used an RTX 4090 with torch 2.13.0, transformers 5.15.0 and accelerate 1.14.0 (Qwen3-14B: L40S, torch 2.9.1, transformers 5.1.0).
- **Sampled seeds** (`seed_robustness`): budgets 128, 384 and 768; seeds 1, 7, 21, 42 and 99; temperature 0.6; top_p 0.95.
- **Pinned revisions:**
  - Phi `0e3b1e2d…`
  - Qwen3-4B `1cfa9a72…`
  - DeepSeek-8B `6a6f4aa4…`
  - Qwen3-8B `b968826d…`
  - Qwen3-14B `40c06982…`
  - Granite-8B `610d8c6e…`

## Analysis entry points (CPU)

- `analysis/extract5.py` is the 2% matcher.
- `analysis/revision/extract8.py` produces the reader segments.
- `analysis/cpu_revisions/cr_common.py` (via the head of `t10b`) does hybrid values, categories and the family bootstrap.
- `analysis/rev25/*` covers the readout parser, audit, regression and robustness.
- `analysis/followup/prep_traces.py` runs the extractors on new JSONL files.

## What AuctionNet changes

- **Data source:** the competing price per impression, rather than an eBay closing price.
- **Products:** see `DATA_LICENSE.md` and `auctionnet_runner/README.md`.
- **Split:** by delivery period, rather than random.
- **Bid step:** chosen per product.
- **Generation:** vLLM, batched, for traces only. Prompt ids, decoding, letter readout and scoring are exactly the eBay code, run in the eBay environments.
