# License status and third-party assets

The original software written for this project is released under the MIT
License in `LICENSE`. That license covers the runner, prompt construction,
analysis, plotting, validation, and packaging source code. It does not relicense
third-party checkpoints, source data, model-generated text, the ACM LaTeX
class, or other third-party material.

Third-party checkpoints are not included. Users obtain them from their original
hosts and must follow their licenses:

- Qwen3-4B, Qwen3-8B, and Qwen3-14B: Apache-2.0
  (`https://huggingface.co/Qwen/Qwen3-4B`,
  `https://huggingface.co/Qwen/Qwen3-8B`,
  `https://huggingface.co/Qwen/Qwen3-14B`);
- Microsoft Phi-4-mini-reasoning: MIT
  (`https://huggingface.co/microsoft/Phi-4-mini-reasoning`);
- DeepSeek-R1-Distill-Llama-8B: DeepSeek's terms plus the applicable Llama 3.1
  Community License and acceptable-use policy for the base model.
- IBM Granite-3.2-8B-Instruct: Apache-2.0
  (`https://huggingface.co/ibm-granite/granite-3.2-8b-instruct`).

The ACM `acmart` class and bibliography style in `www2027/paper/` are
distributed by ACM for paper preparation.

AuctionNet (used in `www2027/`) is simulated data released under Apache-2.0 by
its authors (Su et al., 2024). Its raw files are not included; only derived
scenario files and model decisions are, with attribution.

The Online Auctions Kaggle mirror reports an unknown license. The raw dataset is
excluded. The archive includes the exact transformed split and scenario
artifacts required to audit the prompts and held-out scoring; they are supplied
only for scholarly verification, retain the source provenance and license
uncertainty, and are not relicensed under MIT. Users can instead reconstruct
them from the cited source with the provided script; see `data/README.md`.

Saved JSONL files contain model-generated responses and transformed scenario
attributes for result verification; they do not contain source auction rows or
bidder identifiers. Their inclusion does not assert a new license over model
outputs or the underlying source dataset.
