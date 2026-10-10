.PHONY: help check reproduce paper package
PYTHON ?= $(if $(REPRO_PYTHON_BIN),$(REPRO_PYTHON_BIN),python3)
help:
	@echo 'make check       Verify the current release and run the AI4F CPU audits'
	@echo 'make reproduce   Replay the complete preserved experiment archive (CPU)'
	@echo 'make paper       Compile the current anonymous AI4F source into build/'
	@echo 'make package     Rebuild the compact AI4F download'
check:
	"$(PYTHON)" scripts/check_release.py
	REPRO_PYTHON_BIN="$(PYTHON)" bash reproduce_ai4f_workshop.sh
	"$(PYTHON)" validate_repository.py
reproduce:
	REPRO_PYTHON_BIN="$(PYTHON)" bash reproduce_cpu_revision.sh
paper:
	$(MAKE) -C ai4f_workshop all PYTHON="$(PYTHON)"
	$(MAKE) -C ai4f_workshop audit PYTHON="$(PYTHON)"
package:
	"$(PYTHON)" scripts/package_ai4f.py
