.PHONY: all clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	mkdir -p output/pdf
	cp main.pdf output/pdf/reasoning_compute_arms_races_econml_2026.pdf

clean:
	latexmk -C
