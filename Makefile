.PHONY: help check reproduce paper
PYTHON ?= $(if $(REPRO_PYTHON_BIN),$(REPRO_PYTHON_BIN),python3)
help:
	@echo 'make check       Validate the repository and rebuild key paper numbers from saved decisions (CPU)'
	@echo 'make reproduce   Replay the complete preserved eBay experiment archive (CPU)'
	@echo 'make paper       Compile the paper in www2027/paper/'
check:
	"$(PYTHON)" validate_repository.py
	bash www2027/setup_paths.sh
	cd /tmp/www2027/analysis/auctionnet && "$(PYTHON)" an_overnight_analysis.py
	cd /tmp/www2027/analysis/arxiv && "$(PYTHON)" make_tables.py
reproduce:
	REPRO_PYTHON_BIN="$(PYTHON)" bash reproduce_cpu_revision.sh
paper:
	cd www2027/paper && pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main
