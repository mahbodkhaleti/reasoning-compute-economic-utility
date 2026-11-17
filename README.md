# Budget, Readout or Reasoning? Where Small Reasoning Models Lose Value in Auction Bidding

Code, saved model decisions and analyses for the paper. Six open reasoning models (Phi-4-mini-reasoning, Qwen3-4B, -8B and -14B, DeepSeek-R1-Distill-Llama-8B and Granite-3.2-8B-Instruct) choose a bid from a bid landscape. Win curves come from eBay closing prices and from AuctionNet, a simulated advertising market. Bids are scored on held-out auctions against an exact calculator, and every reasoning trace is parsed by code to find where the gap to the calculator opens.

**Start with [`www2027/README.md`](www2027/README.md).** It describes the paper's folder and maps every result in the paper to the script that produces it and the file it writes. The paper source and PDF are in [`www2027/paper/`](www2027/paper/).

## Quick start

No GPU is needed to reproduce the paper's numbers from the saved decisions.

```sh
bash www2027/setup_paths.sh
pip install numpy pandas scipy statsmodels matplotlib
make check
```

`make check` validates the repository (file hashes, preserved outputs, scenario files) and rebuilds the AuctionNet summary and the full set of result tables from the saved decisions.

| Task | Command | Requirements |
| --- | --- | --- |
| Validate and rebuild key results | `make check` | Python 3.10+; no GPU |
| Replay the complete preserved eBay archive | `make reproduce` | Python 3.13 and `offline_replay/requirements.txt`; no GPU |
| Compile the paper | `make paper` | LaTeX with the ACM `acmart` class |

See the [reproduction guide](docs/REPRODUCE.md) for details.

## Repository guide

| Area | Contents |
| --- | --- |
| [www2027/](www2027/README.md) | The paper, its analysis code, results, saved decisions from the follow-up and AuctionNet runs, and the GPU runners |
| [preserved_outputs/](preserved_outputs/) | Saved model outputs from the original eBay runs |
| [robustness_extensions/](robustness_extensions/) | Runner and analysis code for the original eBay runs, with server outputs |
| [new_models_full_server_package/](new_models_full_server_package/), [qwen3_14b_addition/](qwen3_14b_addition/) | Run packages for Granite-8B and Qwen3-14B, with transformed scenarios |
| [synthetic_ipv_benchmark/](synthetic_ipv_benchmark/), [synthetic_existing4_server_package/](synthetic_existing4_server_package/) | The synthetic first-price task and its runs |
| [offline_replay/](offline_replay/README.md) | CPU replay of the preserved archive |
| [cpu_revision/](cpu_revision/), [decision_audit/](decision_audit/) | Saved-data analyses: uncertainty, calculator comparisons, decision checks |
| [environment/](environment/), [data/](data/README.md) | Recorded inference environment; data acquisition and provenance |

The [repository map](docs/REPOSITORY_MAP.md) describes each folder in more detail.

## Data and licences

The raw eBay auction data must be obtained separately under the source provider's terms ([data/README.md](data/README.md)); transformed scenarios and saved outputs are included. AuctionNet is simulated data released under Apache-2.0; its raw files are downloaded by the runner and only derived scenarios and decisions are included. No model weights are distributed. Code is MIT-licensed; see [LICENSES.md](LICENSES.md) for third-party terms.
