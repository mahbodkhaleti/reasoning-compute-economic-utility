# License status and third-party assets

The original software written for this project is released under the MIT
License in `LICENSE`. That license covers the runner, prompt construction,
analysis, plotting, validation, and packaging source code. It does not relicense
third-party checkpoints, source data, model-generated text, the NeurIPS style,
or other third-party material.

Third-party checkpoints are not included. Users obtain them from their original
hosts and must follow their licenses:

- Qwen3-4B, Qwen3-8B, and the optional Qwen3-14B extension: Apache-2.0
  (`https://huggingface.co/Qwen/Qwen3-4B`,
  `https://huggingface.co/Qwen/Qwen3-8B`,
  `https://huggingface.co/Qwen/Qwen3-14B`);
- Microsoft Phi-4-mini-reasoning: MIT
  (`https://huggingface.co/microsoft/Phi-4-mini-reasoning`);
- DeepSeek-R1-Distill-Llama-8B: DeepSeek's terms plus the applicable Llama 3.1
  Community License and acceptable-use policy for the base model.

The NeurIPS style in `paper/neurips_2026.sty` is the official submission style
distributed for paper preparation.

The Online Auctions Kaggle mirror reports an unknown license. The raw dataset
and processed auction-derived CSVs are excluded. Users obtain the source from
its cited host and reconstruct those files locally with the provided script;
see `data/README.md`.

Saved JSONL files contain model-generated responses and transformed scenario
attributes for result verification; they do not contain source auction rows or
bidder identifiers. Their inclusion does not assert a new license over model
outputs or the underlying source dataset.
