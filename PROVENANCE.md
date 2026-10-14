# Synthetic provenance

Analytic geometry, seed 20260901, 109 Rician magnitude volumes; no participant data.

The primary correction and shared-baseline classifier use the unchanged numerical function bodies recorded in `mean_domain_core.provenance.json`. All volumes, including b=0, are corrected after actual MP-PCA. Scanner distortions, N4 bias fields and drift are not imposed here; their factors are identity.

LV/SA are primary regions. Other is a separate same-rule challenge, outside regional estimates. HDBSCAN does not recalibrate the classifier.

Reference recovered LV-SA difference: 2.453034651846 deg C; matched quadratic control: 2.457495453560 deg C. Full software/input/code/array hashes accompany the expected outputs.
