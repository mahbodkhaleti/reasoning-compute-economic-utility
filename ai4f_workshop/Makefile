.PHONY: all check audit clean
PYTHON ?= python3
all:
	latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex
check:
	$(PYTHON) check_method_bounds.py
	$(PYTHON) cpu_reference/test_reference.py
	$(PYTHON) cpu_reference/check_saved_protocol.py
audit:
	$(PYTHON) verify_numbers.py --built --no-write
clean:
	latexmk -outdir=build -C main.tex
