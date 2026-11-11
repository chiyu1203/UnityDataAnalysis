Locust looming-stimulus censored model analysis
================================================

Observations: 782
Observed slowing events: 501
Right-censored trials: 281
Fraction censored: 0.359
Animals: 41
Colours: black, gregarious
Speeds: 1.0, 4.0
Trial-order adjustment: True

Candidate Bayesian AFT models:
  null         = neither colour nor speed
  speed        = speed only
  colour       = colour only
  additive     = colour + speed
  interaction  = colour * speed

Highest-ranked Bayesian PSIS-LOO model: speed

Statsmodels MixedLM was intentionally skipped because censored trials
are present. MixedLM cannot encode right-censoring and would treat the
censoring time as if it were the true slowing time.

Posterior hypothesis-region probabilities:
  interaction: 0.8765
  speed_plus_colour_additive: 0.0866
  speed_only: 0.0369
  colour_only: 0.0000
  null_or_negligible: 0.0000

ROPE: effects below approximately 5.00% multiplicative difference in latency are treated as biologically negligible.

Interpretation:
  - time ratio < 1 means earlier slowing for the numerator level.
  - time ratio > 1 means later slowing for the numerator level.
  - a censored row contributes the probability that its true latency
    exceeds the censoring time, rather than being treated as an event.
  - LOO stacking weights are predictive weights, not literal posterior
    probabilities that a biological hypothesis is true.
  - If non-response is a qualitatively separate 'never responds' state,
    consider a two-part/hurdle or cure-fraction model instead.

Key output files:
  bayesian_loo_model_comparison.csv
  bayesian_hypothesis_region_probabilities.csv
  bayesian_effect_ratios.csv
  bayesian_loo_model_comparison.png