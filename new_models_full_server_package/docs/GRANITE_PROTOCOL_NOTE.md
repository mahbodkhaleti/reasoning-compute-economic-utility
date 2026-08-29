# Granite protocol note

The exact evaluated checkpoint is `ibm-granite/granite-3.2-8b-instruct` at revision `610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695`. No Granite checkpoint may be substituted without creating a new result set and manifest.

The pinned tokenizer's chat template supports `thinking=True`; it does not expose Qwen's `enable_thinking=True` interface. The runner therefore invokes Granite with its documented `thinking` template argument, externally generates exactly the assigned number of intermediate token IDs, saves that text, and then applies the same constrained option-letter readout and scoring code used for Qwen. For zero compute it generates no intermediate trace.

This is the closest technically valid implementation, but it is recorded as `reasoning_protocol_matched=false`: Granite's prompt/template-based thought-process mode is not claimed to be semantically identical to Qwen3's native thinking mode. Granite results should consequently be described as cross-family protocol sensitivity. The runner fails if the pinned template rejects the `thinking` argument; it never silently falls back to a different prompt.
