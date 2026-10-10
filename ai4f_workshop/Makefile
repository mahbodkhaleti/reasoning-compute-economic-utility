.PHONY: all check clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main_ai4f_v4.tex
check:
	python3 check_method_bounds.py
	python3 cpu_reference/test_reference.py
clean:
	latexmk -C main_ai4f_v4.tex
