PY ?= python3
export MPLBACKEND ?= Agg
export MPLCONFIGDIR ?= /tmp/csf-thermometry-matplotlib
export XDG_CACHE_HOME ?= /tmp/csf-thermometry-cache
SYNTHETIC_BIDS ?= $(CURDIR)/synthetic_bids
REPRODUCED_OUT ?= $(CURDIR)/reproduced_outputs
REPRODUCED_FIGURE ?= $(CURDIR)/reproduced_figure
PREPROCESSING ?= $(CURDIR)/preprocessing

synthetic:
	$(PY) -m reproducibility.code.synthetic.generate_synthetic_bids --output "$(SYNTHETIC_BIDS)" --force

analysis:
	$(PY) -m reproducibility.code.synthetic.run_synthetic_example --bids-root "$(SYNTHETIC_BIDS)" --output-dir "$(REPRODUCED_OUT)/sub-synthetic01" --preprocessing-dir "$(PREPROCESSING)" --no-plot

control:
	$(PY) -m reproducibility.code.synthetic.run_synthetic_example --bids-root "$(SYNTHETIC_BIDS)" --output-dir "$(REPRODUCED_OUT)/quadratic_control/sub-synthetic01" --preprocessing-dir "$(PREPROCESSING)" --branch quadratic_control --no-plot

figure:
	$(PY) -m reproducibility.code.synthetic.plot_synthetic_validation --bids-root "$(SYNTHETIC_BIDS)" --analysis-root "$(REPRODUCED_OUT)/sub-synthetic01" --output-dir "$(REPRODUCED_FIGURE)"

test:
	$(PY) -m pytest -q tests

validate:
	bids-validator "$(SYNTHETIC_BIDS)" --datasetTypes raw
	bids-validator "$(SYNTHETIC_BIDS)/derivatives/thermometrysynthetic" --datasetTypes derivative

.PHONY: synthetic analysis control figure test validate
