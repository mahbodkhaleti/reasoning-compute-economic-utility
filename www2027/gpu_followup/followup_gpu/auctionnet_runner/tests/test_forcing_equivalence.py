"""GPU test: for 20 decisions x 2 models, greedy forced traces generated directly at 256 and 512 tokens must equal the
first 256 / 512 tokens of the 768-token trace (the prefix reuse assumption). Skipped without a GPU."""
import os, subprocess, sys, json, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _gpu():
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False

@pytest.mark.skipif(not _gpu(), reason="needs a GPU and env_v")
@pytest.mark.parametrize("model", ["qwen3_4b", "deepseek_r1_llama_8b"])
def test_prefix_reuse(model):
    subprocess.run([os.path.join(ROOT, "env_a", "bin", "python"), "-m", "auctionnet_runner.run", "run", "--dataset", "ebay",
                    "--model", model, "--run-type", "prefix_test"], cwd=ROOT, check=True)
    rows = [json.loads(x) for x in open(os.path.join(ROOT, "results", "extra", "prefix_test.jsonl"))]
    r = [x for x in rows if x["model"] == model][-1]
    assert r["same256"] == r["n"] and r["same512"] == r["n"], r
