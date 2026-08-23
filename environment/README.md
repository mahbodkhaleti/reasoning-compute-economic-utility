# Environment

`recorded-server-environment.txt` contains every exact version recovered from
the inference server. `robustness_extensions/scripts/requirements.txt` is the
original install specification and contains lower bounds for packages whose
historical exact versions were not captured.

The three exact Python package versions recovered after the run are
Transformers 5.15.0, Accelerate 1.14.0, and PyTorch 2.13.0+cu130. A complete
historical `pip freeze` was not saved, so this repository does not pretend that
the remaining transitive dependency versions are known.

If the original server environment still exists, archive it before changing it:

```bash
/path/to/original/server/venv/bin/pip freeze > environment/pip-freeze-server.txt
```

The `pynvml` deprecation warning seen during inspection is operational rather
than numerical. It does not alter the recorded results.
