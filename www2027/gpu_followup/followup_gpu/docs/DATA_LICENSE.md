# AuctionNet data: licence and handling

- **Licence.** The AuctionNet repository is distributed under Apache License 2.0 (`LICENSE.txt`). The dataset readme in `pre_generated_dataset/readme_dataset.md` gives download links but no separate data terms.
- **Nature of the data.** It is *simulated*. Ad opportunities come from a deep generative model, and 48 auto-bidding agents compete in a 3-slot GSP auction (`simul_bidding_env/Environment/BiddingEnv.py`: `NUM_SLOTS = 3`, reserve 0.01, slot exposure 1/0.8/0.6). The paper must not call these real auction logs.
- **Raw files.** They are not committed. `auctionnet_runner/data.py` downloads them on the GPU or data machine.
- **Derived files.** We commit only the derived scenario file (bid menus with estimated and held-out win probabilities) and the per-product empirical win curves. These are aggregates of Apache-2.0 material, released with attribution to AuctionNet (Su et al., NeurIPS 2024 Datasets and Benchmarks).
- **Conclusion.** Committing derived scenarios is allowed under Apache 2.0, with attribution and a notice. Redistributing the raw zip files is avoided out of caution, because the dataset terms are not stated separately.
