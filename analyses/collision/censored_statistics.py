#!/usr/bin/env python3
"""
Locust looming-stimulus censored repeated-measures analysis
============================================================

Compares five hypotheses/models for slowing latency:

    null         : no colour or speed effect
    speed        : speed effect only
    colour       : colour effect only
    additive     : independent colour + speed effects
    interaction  : colour + speed + colour-by-speed interaction

Designed for:
    - repeated trials within locusts
    - 3 colours (and in practice any >=2 colours)
    - exactly 2 speeds
    - positive slowing-latency/censoring times
    - optional right-censoring for trials with no observed slowing response

Frequentist analysis:
    statsmodels MixedLM with a random intercept for locust.
    Models are fit using ML (REML=False) so fixed-effect structures can
    be compared with AIC/BIC and likelihood-ratio tests.

Bayesian analysis:
    PyMC hierarchical right-censored LogNormal accelerated-failure-time (AFT)
    model with a locust random intercept.
    Colour effects and colour-by-speed interactions use sum-to-zero
    parameterizations, so no colour is treated as the privileged
    biological baseline.
    Models are compared with PSIS-LOO using ArviZ.

Optional trial-order adjustment:
    If a trial-order column exists, it is included in EVERY candidate
    model. This helps separate stimulus effects from general
    habituation/sensitization across repeated trials.

IMPORTANT:
    1. slowing_time must be > 0.
    2. Trials with no slowing before the observation window ends are
       retained as right-censored observations (event_observed = 0).
    3. If response times can legitimately be <= 0 (for example time
       relative to predicted collision), replace the LogNormal model
       with a suitable Normal/Student-t hierarchical model.

Example input CSV
-----------------
locust_id,trial_order,colour,speed,slowing_time,event_observed
L01,1,colour1,speed1,1.42,1
L01,2,colour2,speed1,1.31,1
L01,3,colour3,speed1,2.00,0
L01,4,colour1,speed2,0.82,1
L01,5,colour2,speed2,0.79,1
L01,6,colour3,speed2,2.00,0

For event_observed=0, slowing_time is the censoring time. For example,
2.00 means the animal was observed for 2.00 time units without slowing;
the latent slowing time is known only to be > 2.00.

If colour and speed are combined in a condition column such as
"colour1_speed2", the script can split it automatically from the final
underscore.

Usage from Python
-----------------
Minimal:

    if __name__ == "__main__":
        csv_file = "out_censored.csv"
        main(csv_file)

With optional settings:

    if __name__ == "__main__":
        main(
            "out_censored.csv",
            output_dir="results",
            draws=2000,
            tune=2000,
            chains=4,
            cores=4,
            rope_percent=5.0,
        )

Custom column names can also be supplied directly to main().

Dependencies
------------
    numpy
    pandas
    scipy
    matplotlib
    statsmodels
    pymc>=6.4
    arviz>=1.0

Install, for example:
    pip install numpy pandas scipy matplotlib statsmodels pymc arviz

ArviZ compatibility
-------------------
This script supports both legacy ArviZ 0.x and the ArviZ 1.x API.
ArViZ 1.x renamed summary credible-interval arguments, changed hdi(),
and removed the `ic` argument from compare() because model comparison
uses PSIS-LOO directly.
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# PyMC and ArviZ are imported lazily so command-line help and the
# frequentist setup can still be inspected if the Bayesian stack has a
# local version conflict.
pm = None
az = None

def ensure_bayesian_imports():
    global pm, az
    if pm is not None and az is not None:
        return
    try:
        import arviz as _az
        import pymc as _pm
    except Exception as exc:
        raise RuntimeError(
            'Could not import a compatible PyMC/ArviZ stack. '
            'Install or upgrade them together, e.g. '
            '`pip install --upgrade pymc arviz`. Original error: '
            f'{exc}'
        ) from exc
    az = _az
    pm = _pm


def arviz_summary_compat(data, var_names, prob=0.95):
    """Call az.summary across ArviZ <1 and >=1 APIs, requesting an HDI."""
    ensure_bayesian_imports()
    params = inspect.signature(az.summary).parameters
    kwargs = {"var_names": var_names}

    if "ci_prob" in params:
        # ArviZ >= 1.0
        kwargs["ci_prob"] = prob
        if "ci_kind" in params:
            kwargs["ci_kind"] = "hdi"
    elif "hdi_prob" in params:
        # ArviZ < 1.0
        kwargs["hdi_prob"] = prob

    return az.summary(data, **kwargs)


def arviz_compare_compat(compare_dict, method="stacking"):
    """Compare models with PSIS-LOO across ArviZ <1 and >=1 APIs."""
    ensure_bayesian_imports()
    params = inspect.signature(az.compare).parameters
    kwargs = {"method": method}

    # ArviZ < 1.0 allowed LOO/WAIC selection through `ic`.
    # ArviZ >= 1.0 compares with PSIS-LOO directly and removed `ic`.
    if "ic" in params:
        kwargs["ic"] = "loo"

    return az.compare(compare_dict, **kwargs)


def arviz_hdi_compat(samples, prob=0.95):
    """Compute an HDI across ArviZ <1 and >=1 APIs."""
    ensure_bayesian_imports()
    params = inspect.signature(az.hdi).parameters

    if "prob" in params:
        # ArviZ >= 1.0
        return az.hdi(samples, prob=prob)
    if "hdi_prob" in params:
        # ArviZ < 1.0
        return az.hdi(samples, hdi_prob=prob)

    # Defensive fallback for unusual intermediate versions.
    try:
        return az.hdi(samples, prob=prob)
    except TypeError:
        return az.hdi(samples, hdi_prob=prob)


def save_compare_plot_compat(comparison, filename):
    """Save az.plot_compare output for both legacy and ArviZ 1.x plotting APIs."""
    ensure_bayesian_imports()
    plot_obj = az.plot_compare(comparison)

    # ArviZ >=1 generally returns a PlotCollection with .savefig().
    if hasattr(plot_obj, "savefig"):
        plot_obj.savefig(filename, dpi=180)
    else:
        # Legacy ArviZ generally returns matplotlib Axes.
        plt.tight_layout()
        plt.savefig(filename, dpi=180, bbox_inches="tight")

    plt.close("all")


from scipy.stats import chi2


MODEL_TERMS = {
    "null": set(),
    "speed": {"speed"},
    "colour": {"colour"},
    "additive": {"speed", "colour"},
    "interaction": {"speed", "colour", "interaction"},
}


@dataclass
class AnalysisConfig:
    """Configuration used internally by the analysis functions."""

    csv_file: Path
    output_dir: Path = Path("locust_censored_model_results")
    animal_col: str = "locust_id"
    colour_col: str = "colour"
    speed_col: str = "speed"
    response_col: str = "slowing_time"
    event_col: str = "event_observed"
    censor_time: float | None = None
    trial_col: str | None = "trial_order"
    condition_col: str | None = None
    colour_order: str | None = None
    speed_order: str | None = None
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    cores: int = 4
    target_accept: float = 0.90
    seed: int = 12345
    rope_percent: float = 5.0


def ordered_levels(values: pd.Series, requested: str | None) -> list[str]:
    observed = [str(x) for x in pd.unique(values.dropna())]

    if requested is None:
        return sorted(observed)

    levels = [x.strip() for x in requested.split(",") if x.strip()]
    missing = sorted(set(observed) - set(levels))
    extra = sorted(set(levels) - set(observed))

    if missing:
        raise ValueError(
            f"Requested order omits observed levels: {missing}. "
            f"Observed levels are {sorted(observed)}."
        )
    if extra:
        warnings.warn(
            f"Requested order contains levels not present in the data: {extra}"
        )

    return [x for x in levels if x in observed]


def prepare_data(args: AnalysisConfig) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(args.csv_file)

    # If separate colour/speed columns do not exist, optionally split condition.
    if args.colour_col not in df.columns or args.speed_col not in df.columns:
        if args.condition_col is None:
            raise ValueError(
                f"Could not find both '{args.colour_col}' and '{args.speed_col}'. "
                "Supply --condition-col if colour and speed are stored together."
            )
        if args.condition_col not in df.columns:
            raise ValueError(
                f"Condition column '{args.condition_col}' was not found."
            )

        split = df[args.condition_col].astype(str).str.rsplit("_", n=1, expand=True)
        if split.shape[1] != 2:
            raise ValueError(
                "Could not split the condition column into colour and speed "
                "using the final underscore."
            )
        df[args.colour_col] = split.iloc[:, 0]
        df[args.speed_col] = split.iloc[:, 1]

    required = [args.animal_col, args.colour_col, args.speed_col, args.response_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    response = pd.to_numeric(df[args.response_col], errors="coerce")

    if args.event_col in df.columns:
        event = pd.to_numeric(df[args.event_col], errors="coerce")
    else:
        warnings.warn(
            f"Event column '{args.event_col}' was not found. Assuming all trials "
            "have observed slowing events (no censoring)."
        )
        event = pd.Series(1.0, index=df.index)

    # Validate event coding before dropping rows.
    bad_event = event.notna() & ~event.isin([0, 1])
    if bad_event.any():
        examples = sorted(pd.unique(event[bad_event]).tolist())
        raise ValueError(
            f"'{args.event_col}' must be coded 1=event observed and 0=right-censored. "
            f"Invalid values include: {examples[:10]}"
        )

    if args.censor_time is not None:
        if args.censor_time <= 0:
            raise ValueError("--censor-time must be strictly positive.")
        # Fill only missing response times on known censored rows. Existing values
        # are preserved so row-specific censoring times remain possible.
        fill_mask = (event == 0) & response.isna()
        response.loc[fill_mask] = float(args.censor_time)

    # A censored row needs a known censoring time.
    missing_censor_time = (event == 0) & response.isna()
    if missing_censor_time.any():
        raise ValueError(
            f"{int(missing_censor_time.sum())} censored rows have no censoring time. "
            f"Put the censoring time in '{args.response_col}' or pass censor_time=... to main()."
        )

    dat = pd.DataFrame(
        {
            "_animal": df[args.animal_col],
            "_colour": df[args.colour_col],
            "_speed": df[args.speed_col],
            "_response": response,
            "_event": event,
        }
    )

    use_trial = (
        args.trial_col is not None
        and str(args.trial_col).upper() != "NONE"
        and args.trial_col in df.columns
    )
    if use_trial:
        dat["_trial"] = pd.to_numeric(df[args.trial_col], errors="coerce")

    before = len(dat)
    needed = ["_animal", "_colour", "_speed", "_response", "_event"]
    if use_trial:
        needed.append("_trial")
    dat = dat.dropna(subset=needed).copy()
    dropped = before - len(dat)

    if dat.empty:
        raise ValueError("No complete observations remain after data preparation.")

    dat["_animal"] = dat["_animal"].astype(str)
    dat["_colour"] = dat["_colour"].astype(str)
    dat["_speed"] = dat["_speed"].astype(str)
    dat["_event"] = dat["_event"].astype(int)

    if (dat["_response"] <= 0).any():
        bad_n = int((dat["_response"] <= 0).sum())
        raise ValueError(
            f"{bad_n} event/censoring times are <= 0. The LogNormal AFT model "
            "requires strictly positive elapsed times."
        )

    if dat["_event"].sum() == 0:
        raise ValueError(
            "All trials are censored. At least some observed slowing events are "
            "needed to identify this model from the data."
        )

    colour_levels = ordered_levels(dat["_colour"], args.colour_order)
    speed_levels = ordered_levels(dat["_speed"], args.speed_order)

    if len(colour_levels) < 2:
        raise ValueError("At least two colours are required.")
    if len(speed_levels) != 2:
        raise ValueError(
            f"This script expects exactly 2 speed levels; found {len(speed_levels)}: "
            f"{speed_levels}"
        )

    dat["_colour"] = pd.Categorical(
        dat["_colour"], categories=colour_levels, ordered=True
    )
    dat["_speed"] = pd.Categorical(
        dat["_speed"], categories=speed_levels, ordered=True
    )
    dat["_log_response"] = np.log(dat["_response"])

    if use_trial:
        sd_trial = dat["_trial"].std(ddof=0)
        if not np.isfinite(sd_trial) or sd_trial == 0:
            warnings.warn(
                "Trial-order column has no variation; trial order will be omitted."
            )
            use_trial = False
            dat = dat.drop(columns=["_trial"])
        else:
            dat["_trial_z"] = (
                dat["_trial"] - dat["_trial"].mean()
            ) / sd_trial

    n_events = int(dat["_event"].sum())
    n_censored = int((dat["_event"] == 0).sum())

    metadata = {
        "input_file": str(args.csv_file),
        "n_observations": int(len(dat)),
        "n_events": n_events,
        "n_censored": n_censored,
        "fraction_censored": float(n_censored / len(dat)),
        "n_animals": int(dat["_animal"].nunique()),
        "n_colours": len(colour_levels),
        "colour_levels": colour_levels,
        "speed_levels": speed_levels,
        "trial_order_adjusted": bool(use_trial),
        "rows_dropped_for_missing_values": int(dropped),
    }

    return dat, metadata


def fit_mixedlm_with_retries(formula: str, dat: pd.DataFrame):
    model = smf.mixedlm(
        formula,
        data=dat,
        groups=dat["_animal"],
        re_formula="1",
    )

    last_error = None
    finite_results = []

    # MixedLM can occasionally report convergence at a singular boundary
    # with an infinite log-likelihood under one optimizer. Try several
    # optimizers and retain the best finite converged solution.
    for method in ("lbfgs", "bfgs", "powell", "cg", "nm"):
        try:
            result = model.fit(
                reml=False,
                method=method,
                disp=False,
            )
            if np.isfinite(result.llf) and np.isfinite(result.aic):
                finite_results.append(result)
        except Exception as exc:
            last_error = exc

    converged = [r for r in finite_results if bool(r.converged)]
    candidates = converged if converged else finite_results

    if candidates:
        return max(candidates, key=lambda r: float(r.llf))

    raise RuntimeError(
        f"MixedLM failed to produce a finite likelihood for formula {formula!r}. "
        f"Last error: {last_error}. This can occur when the random-intercept "
        "variance is estimated on the boundary (approximately zero)."
    )


def likelihood_ratio_test(reduced, full) -> dict:
    lr = 2.0 * (full.llf - reduced.llf)
    df_diff = len(full.fe_params) - len(reduced.fe_params)

    if df_diff <= 0:
        return {"LR": np.nan, "df": df_diff, "p": np.nan}

    # Guard against tiny negative LR values from numerical fitting noise.
    lr_for_p = max(float(lr), 0.0)
    return {
        "LR": float(lr),
        "df": int(df_diff),
        "p": float(chi2.sf(lr_for_p, df_diff)),
    }


def run_statsmodels(dat: pd.DataFrame, use_trial: bool, output_dir: Path):
    trial_term = " + _trial_z" if use_trial else ""

    formulas = {
        "null": "_log_response ~ 1" + trial_term,
        "speed": "_log_response ~ C(_speed)" + trial_term,
        "colour": "_log_response ~ C(_colour)" + trial_term,
        "additive": "_log_response ~ C(_speed) + C(_colour)" + trial_term,
        "interaction": "_log_response ~ C(_speed) * C(_colour)" + trial_term,
    }

    results = {}

    print("\n" + "=" * 78)
    print("STATSMODELS MIXED-EFFECTS MODELS")
    print("=" * 78)

    for name, formula in formulas.items():
        print(f"\nFitting {name:>11s}: {formula}")
        result = fit_mixedlm_with_retries(formula, dat)
        results[name] = result
        print(
            f"  converged={result.converged}  "
            f"logLik={result.llf:.3f}  AIC={result.aic:.3f}  BIC={result.bic:.3f}"
        )

        # Save each detailed summary.
        with open(output_dir / f"statsmodels_{name}_summary.txt", "w") as f:
            f.write(result.summary().as_text())

    comparison = pd.DataFrame(
        {
            name: {
                "logLik": res.llf,
                "AIC": res.aic,
                "BIC": res.bic,
                "n_fixed_effects": len(res.fe_params),
                "converged": bool(res.converged),
            }
            for name, res in results.items()
        }
    ).T

    comparison["delta_AIC"] = comparison["AIC"] - comparison["AIC"].min()
    aic_support = np.exp(-0.5 * comparison["delta_AIC"].astype(float))
    comparison["Akaike_weight"] = aic_support / aic_support.sum()
    comparison = comparison.sort_values("AIC")

    comparison.to_csv(output_dir / "statsmodels_model_comparison.csv")

    lrt_spec = {
        "speed_vs_null": ("null", "speed"),
        "colour_vs_null": ("null", "colour"),
        "add_speed_given_colour": ("colour", "additive"),
        "add_colour_given_speed": ("speed", "additive"),
        "interaction_vs_additive": ("additive", "interaction"),
    }

    lrt_rows = {}
    for label, (reduced_name, full_name) in lrt_spec.items():
        lrt_rows[label] = likelihood_ratio_test(
            results[reduced_name],
            results[full_name],
        )

    lrt = pd.DataFrame(lrt_rows).T
    lrt.to_csv(output_dir / "statsmodels_likelihood_ratio_tests.csv")

    print("\nStatsmodels model comparison:")
    print(comparison.to_string())

    print("\nNested likelihood-ratio tests:")
    print(lrt.to_string())

    return results, comparison, lrt


def build_and_sample_pymc(
    dat: pd.DataFrame,
    metadata: dict,
    fixed_terms: set[str],
    args: AnalysisConfig,
):
    ensure_bayesian_imports()
    colour_levels = metadata["colour_levels"]
    speed_levels = metadata["speed_levels"]
    n_colours = len(colour_levels)
    n_speeds = len(speed_levels)

    colour_idx = dat["_colour"].cat.codes.to_numpy(dtype=int)
    speed_idx = dat["_speed"].cat.codes.to_numpy(dtype=int)
    animal_idx, animal_names = pd.factorize(dat["_animal"], sort=True)

    y = dat["_response"].to_numpy(dtype=float)
    event = dat["_event"].to_numpy(dtype=int)
    trial_z = (
        dat["_trial_z"].to_numpy(dtype=float)
        if metadata["trial_order_adjusted"]
        else None
    )

    coords = {
        "animal": [str(x) for x in animal_names],
        "colour": colour_levels,
        "speed": speed_levels,
    }

    # Use observed events to centre the intercept prior. Censored times are
    # observation limits, not realised event times.
    observed_event_times = y[event == 1]
    alpha_centre = float(np.log(np.median(observed_event_times)))

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal(
            "alpha",
            mu=alpha_centre,
            sigma=1.0,
        )

        # Non-centred random intercept for repeated observations within locust.
        sigma_animal = pm.HalfNormal("sigma_animal", sigma=1.0)
        z_animal = pm.Normal("z_animal", 0.0, 1.0, dims="animal")
        animal_effect = pm.Deterministic(
            "animal_effect",
            z_animal * sigma_animal,
            dims="animal",
        )

        mu = alpha + animal_effect[animal_idx]

        if metadata["trial_order_adjusted"]:
            beta_trial = pm.Normal("beta_trial", mu=0.0, sigma=0.5)
            mu = mu + beta_trial * trial_z

        # Sum-to-zero speed main effects.
        if "speed" in fixed_terms:
            speed_raw = pm.Normal(
                "beta_speed_raw", mu=0.0, sigma=0.5, dims="speed"
            )
            beta_speed = pm.Deterministic(
                "beta_speed",
                speed_raw - pm.math.sum(speed_raw) / n_speeds,
                dims="speed",
            )
            mu = mu + beta_speed[speed_idx]

        # Sum-to-zero colour main effects.
        if "colour" in fixed_terms:
            colour_raw = pm.Normal(
                "beta_colour_raw", mu=0.0, sigma=0.5, dims="colour"
            )
            beta_colour = pm.Deterministic(
                "beta_colour",
                colour_raw - pm.math.sum(colour_raw) / n_colours,
                dims="colour",
            )
            mu = mu + beta_colour[colour_idx]

        # Double-centred colour x speed interaction.
        if "interaction" in fixed_terms:
            interaction_raw = pm.Normal(
                "beta_interaction_raw",
                mu=0.0,
                sigma=0.4,
                dims=("colour", "speed"),
            )
            row_mean = pm.math.sum(
                interaction_raw, axis=1, keepdims=True
            ) / n_speeds
            col_mean = pm.math.sum(
                interaction_raw, axis=0, keepdims=True
            ) / n_colours
            grand_mean = pm.math.sum(interaction_raw) / (n_colours * n_speeds)

            beta_interaction = pm.Deterministic(
                "beta_interaction",
                interaction_raw - row_mean - col_mean + grand_mean,
                dims=("colour", "speed"),
            )
            mu = mu + beta_interaction[colour_idx, speed_idx]

        sigma = pm.HalfNormal("sigma", sigma=1.0)

        if metadata["n_censored"] > 0:### different speed should have different upper
            # For an observed event, upper=+inf so y contributes the ordinary
            # LogNormal density. For a censored trial, upper=y, so y contributes
            # P(T >= y), the right-tail/survival probability.
            upper = np.where(event == 0, y, np.inf)
            latent_time = pm.LogNormal.dist(mu=mu, sigma=sigma)
            pm.Censored(
                "slowing_time",
                latent_time,
                lower=None,
                upper=upper,
                observed=y,
            )
        else:
            pm.LogNormal(
                "slowing_time",
                mu=mu,
                sigma=sigma,
                observed=y,
            )

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=min(args.cores, args.chains),
            random_seed=args.seed,
            nuts={"target_accept": args.target_accept},
            return_inferencedata=True,
        )

        idata = pm.compute_log_likelihood(
            idata,
            model=model,
            extend_inferencedata=True,
        )

    return model, idata


def summarise_hypothesis_regions(
    idata,
    metadata: dict,
    rope_percent: float,
    output_dir: Path,
):
    """
    Convert the full interaction-model posterior into five mutually exclusive
    biological regions.

    Main effects are defined from sum-to-zero coefficients:
      speed effect: difference between the two speed main effects
      colour effect: maximum pairwise difference among colour main effects

    Interaction is defined as heterogeneity of the speed contrast across
    colours:
      speed contrast at each colour =
          main speed contrast
          + interaction[colour, speed2]
          - interaction[colour, speed1]

      interaction magnitude =
          max(speed contrast across colours)
          - min(speed contrast across colours)

    A log-ratio ROPE threshold is used to decide whether effects are
    biologically meaningful.
    """
    post = idata.posterior

    beta_speed = np.asarray(post["beta_speed"])
    beta_colour = np.asarray(post["beta_colour"])
    beta_interaction = np.asarray(post["beta_interaction"])

    # Flatten chain/draw, retain factor dimensions.
    beta_speed = beta_speed.reshape(-1, beta_speed.shape[-1])
    beta_colour = beta_colour.reshape(-1, beta_colour.shape[-1])
    beta_interaction = beta_interaction.reshape(
        -1,
        beta_interaction.shape[-2],
        beta_interaction.shape[-1],
    )

    # Exactly two speeds are required by prepare_data().
    speed_main_contrast = beta_speed[:, 1] - beta_speed[:, 0]

    # Overall colour effect: largest pairwise colour main-effect contrast.
    colour_main_range = (
        np.max(beta_colour, axis=1) - np.min(beta_colour, axis=1)
    )

    # Speed contrast separately at each colour.
    speed_contrast_by_colour = (
        speed_main_contrast[:, None]
        + beta_interaction[:, :, 1]
        - beta_interaction[:, :, 0]
    )

    # Interaction = degree to which the speed contrast changes by colour.
    interaction_range = (
        np.max(speed_contrast_by_colour, axis=1)
        - np.min(speed_contrast_by_colour, axis=1)
    )

    rope = float(np.log(1.0 + rope_percent / 100.0))

    speed_present = np.abs(speed_main_contrast) > rope
    colour_present = colour_main_range > rope
    interaction_present = interaction_range > rope

    categories = {
        "speed_only": speed_present & ~colour_present & ~interaction_present,
        "colour_only": colour_present & ~speed_present & ~interaction_present,
        "null_or_negligible": ~speed_present & ~colour_present & ~interaction_present,
        "speed_plus_colour_additive": speed_present
        & colour_present
        & ~interaction_present,
        "interaction": interaction_present,
    }

    region_probs = pd.Series(
        {name: float(np.mean(mask)) for name, mask in categories.items()},
        name="posterior_probability",
    ).sort_values(ascending=False)

    region_probs.to_csv(
        output_dir / "bayesian_hypothesis_region_probabilities.csv",
        header=True,
    )

    return (
        region_probs,
        speed_main_contrast,
        beta_colour,
        speed_contrast_by_colour,
        interaction_range,
        rope,
    )


def hdi_summary(samples: np.ndarray, label: str) -> dict:
    ensure_bayesian_imports()
    samples = np.asarray(samples).reshape(-1)
    hdi = np.asarray(arviz_hdi_compat(samples, prob=0.95))
    return {
        "effect": label,
        "median_log_effect": float(np.median(samples)),
        "hdi_2.5_log": float(hdi[0]),
        "hdi_97.5_log": float(hdi[1]),
        "median_ratio": float(np.exp(np.median(samples))),
        "hdi_2.5_ratio": float(np.exp(hdi[0])),
        "hdi_97.5_ratio": float(np.exp(hdi[1])),
        "p_effect_gt_0": float(np.mean(samples > 0)),
        "p_effect_lt_0": float(np.mean(samples < 0)),
    }


def posterior_effect_tables(
    idata,
    metadata: dict,
    output_dir: Path,
):
    post = idata.posterior

    beta_speed = np.asarray(post["beta_speed"])
    beta_colour = np.asarray(post["beta_colour"])
    beta_interaction = np.asarray(post["beta_interaction"])

    beta_speed = beta_speed.reshape(-1, beta_speed.shape[-1])
    beta_colour = beta_colour.reshape(-1, beta_colour.shape[-1])
    beta_interaction = beta_interaction.reshape(
        -1,
        beta_interaction.shape[-2],
        beta_interaction.shape[-1],
    )

    colours = metadata["colour_levels"]
    speeds = metadata["speed_levels"]

    rows = []

    # Speed2 / Speed1 averaged across colours (sum-to-zero interaction averages out).
    speed_contrast = beta_speed[:, 1] - beta_speed[:, 0]
    rows.append(
        hdi_summary(
            speed_contrast,
            f"{speeds[1]} / {speeds[0]} (average across colours)",
        )
    )

    # Pairwise average colour contrasts.
    for i, j in combinations(range(len(colours)), 2):
        contrast = beta_colour[:, j] - beta_colour[:, i]
        rows.append(
            hdi_summary(
                contrast,
                f"{colours[j]} / {colours[i]} (average across speeds)",
            )
        )

    # Speed contrast separately within each colour.
    for c, colour_name in enumerate(colours):
        contrast = (
            speed_contrast
            + beta_interaction[:, c, 1]
            - beta_interaction[:, c, 0]
        )
        rows.append(
            hdi_summary(
                contrast,
                f"{speeds[1]} / {speeds[0]} within {colour_name}",
            )
        )

    effect_table = pd.DataFrame(rows).set_index("effect")
    effect_table.to_csv(output_dir / "bayesian_effect_ratios.csv")

    return effect_table


def run_pymc(
    dat: pd.DataFrame,
    metadata: dict,
    args: AnalysisConfig,
    output_dir: Path,
):
    ensure_bayesian_imports()

    print("\n" + "=" * 78)
    print("PYMC HIERARCHICAL MODELS")
    print("=" * 78)

    models = {}
    idatas = {}

    for name, terms in MODEL_TERMS.items():
        print(f"\nFitting Bayesian model: {name}")
        model, idata = build_and_sample_pymc(dat, metadata, terms, args)
        models[name] = model
        idatas[name] = idata

        # Compact diagnostic summary for parameters that exist in this model.
        candidate_vars = [
            "alpha",
            "beta_speed",
            "beta_colour",
            "beta_interaction",
            "beta_trial",
            "sigma_animal",
            "sigma",
        ]
        vars_here = [v for v in candidate_vars if v in idata.posterior]
        summary = arviz_summary_compat(
            idata,
            var_names=vars_here,
            prob=0.95,
        )
        summary.to_csv(output_dir / f"pymc_{name}_posterior_summary.csv")

    print("\nComputing PSIS-LOO comparison...")
    loo_comparison = arviz_compare_compat(
        idatas,
        method="stacking",
    )
    loo_comparison.to_csv(output_dir / "bayesian_loo_model_comparison.csv")
    print(loo_comparison.to_string())

    # Model-comparison plot.
    try:
        save_compare_plot_compat(
            loo_comparison,
            output_dir / "bayesian_loo_model_comparison.png",
        )
    except Exception as exc:
        warnings.warn(f"Could not save LOO comparison plot: {exc}")

    # Full-model biological interpretation.
    (
        region_probs,
        speed_main_contrast,
        beta_colour,
        speed_contrast_by_colour,
        interaction_range,
        rope,
    ) = summarise_hypothesis_regions(
        idatas["interaction"],
        metadata,
        args.rope_percent,
        output_dir,
    )

    effect_table = posterior_effect_tables(
        idatas["interaction"],
        metadata,
        output_dir,
    )

    print(
        f"\nPosterior hypothesis regions "
        f"(ROPE = ±{args.rope_percent:.2f}% multiplicative difference):"
    )
    print(region_probs.to_string())

    print("\nPosterior effect-ratio summaries:")
    print(effect_table.to_string())

    return models, idatas, loo_comparison, region_probs, effect_table


def make_design_summary(dat: pd.DataFrame, output_dir: Path):
    count_table = (
        dat.groupby(["_colour", "_speed"], observed=True)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    #count_table.to_csv(output_dir / "design_trial_counts.csv", index=False)

    animal_condition_counts = (
        dat.groupby(["_animal", "_colour", "_speed"], observed=True)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    animal_condition_counts.to_csv(
        output_dir / "design_trials_per_animal_condition.csv",
        index=False,
    )

    return count_table


def write_readme(
    output_dir: Path,
    metadata: dict,
    stats_comparison,
    lrt,
    loo_comparison: pd.DataFrame,
    region_probs: pd.Series,
    args: AnalysisConfig,
):
    best_loo = str(loo_comparison.index[0])

    lines = [
        "Locust looming-stimulus censored model analysis",
        "=" * 48,
        "",
        f"Observations: {metadata['n_observations']}",
        f"Observed slowing events: {metadata['n_events']}",
        f"Right-censored trials: {metadata['n_censored']}",
        f"Fraction censored: {metadata['fraction_censored']:.3f}",
        f"Animals: {metadata['n_animals']}",
        f"Colours: {', '.join(metadata['colour_levels'])}",
        f"Speeds: {', '.join(metadata['speed_levels'])}",
        f"Trial-order adjustment: {metadata['trial_order_adjusted']}",
        "",
        "Candidate Bayesian AFT models:",
        "  null         = neither colour nor speed",
        "  speed        = speed only",
        "  colour       = colour only",
        "  additive     = colour + speed",
        "  interaction  = colour * speed",
        "",
        f"Highest-ranked Bayesian PSIS-LOO model: {best_loo}",
    ]

    if stats_comparison is None:
        lines += [
            "",
            "Statsmodels MixedLM was intentionally skipped because censored trials",
            "are present. MixedLM cannot encode right-censoring and would treat the",
            "censoring time as if it were the true slowing time.",
        ]
    else:
        lines += [
            "",
            f"Lowest frequentist MixedLM AIC: {str(stats_comparison.index[0])}",
        ]

    lines += [
        "",
        "Posterior hypothesis-region probabilities:",
    ]
    for name, prob in region_probs.items():
        lines.append(f"  {name}: {prob:.4f}")

    lines += [
        "",
        f"ROPE: effects below approximately {args.rope_percent:.2f}% multiplicative "
        "difference in latency are treated as biologically negligible.",
        "",
        "Interpretation:",
        "  - time ratio < 1 means earlier slowing for the numerator level.",
        "  - time ratio > 1 means later slowing for the numerator level.",
        "  - a censored row contributes the probability that its true latency",
        "    exceeds the censoring time, rather than being treated as an event.",
        "  - LOO stacking weights are predictive weights, not literal posterior",
        "    probabilities that a biological hypothesis is true.",
        "  - If non-response is a qualitatively separate 'never responds' state,",
        "    consider a two-part/hurdle or cure-fraction model instead.",
        "",
        "Key output files:",
        "  bayesian_loo_model_comparison.csv",
        "  bayesian_hypothesis_region_probabilities.csv",
        "  bayesian_effect_ratios.csv",
        "  bayesian_loo_model_comparison.png",
    ]

    (output_dir / "README_results.txt").write_text("\n".join(lines))


def main(
    csv_file: str | Path,
    *,
    output_dir: str | Path = "locust_censored_model_results",
    animal_col: str = "locust_id",
    colour_col: str = "colour",
    speed_col: str = "speed",
    response_col: str = "slowing_time",
    event_col: str = "event_observed",
    censor_time: float | None = None,
    trial_col: str | None = "trial_order",
    condition_col: str | None = None,
    colour_order: str | None = None,
    speed_order: str | None = None,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.90,
    seed: int = 12345,
    rope_percent: float = 5.0,
):
    """
    Run the full censored repeated-measures analysis.

    Parameters
    ----------
    csv_file : str or pathlib.Path
        Path to the input CSV file.
    output_dir : str or pathlib.Path, optional
        Directory where result tables and plots will be written.
    animal_col, colour_col, speed_col, response_col, event_col : str
        Input-data column names.
    censor_time : float or None
        Optional common right-censoring time. Used only to fill missing
        response values on rows where event_observed == 0.
    trial_col : str or None
        Trial-order column. Pass None to disable trial-order adjustment.
    condition_col : str or None
        Optional combined condition column, e.g. ``colour1_speed2``.
    colour_order, speed_order : str or None
        Optional comma-separated level order, e.g.
        ``"colour1,colour2,colour3"`` and ``"speed1,speed2"``.
    draws, tune, chains, cores : int
        PyMC sampling settings.
    target_accept : float
        NUTS target acceptance probability.
    seed : int
        Random seed.
    rope_percent : float
        Multiplicative latency difference regarded as biologically negligible.

    Returns
    -------
    dict
        A dictionary containing prepared data, metadata, model-comparison
        tables, posterior hypothesis-region probabilities, and effect summaries.

    Example
    -------
    >>> main("out_censored.csv")
    """
    args = AnalysisConfig(
        csv_file=Path(csv_file),
        output_dir=Path(output_dir),
        animal_col=animal_col,
        colour_col=colour_col,
        speed_col=speed_col,
        response_col=response_col,
        event_col=event_col,
        censor_time=censor_time,
        trial_col=trial_col,
        condition_col=condition_col,
        colour_order=colour_order,
        speed_order=speed_order,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        seed=seed,
        rope_percent=rope_percent,
    )

    if not args.csv_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dat, metadata = prepare_data(args)

    print("=" * 78)
    print("DATA SUMMARY")
    print("=" * 78)
    print(json.dumps(metadata, indent=2))

    print("\nTrials by colour and speed:")
    count_table = make_design_summary(dat, args.output_dir)
    print(count_table.to_string(index=False))

    censor_table = (
        dat.groupby(["_colour", "_speed"], observed=True)["_event"]
        .agg(n_trials="size", n_events="sum")
        .reset_index()
    )
    censor_table["n_censored"] = censor_table["n_trials"] - censor_table["n_events"]
    censor_table["fraction_censored"] = (
        censor_table["n_censored"] / censor_table["n_trials"]
    )
    censor_table.to_csv(args.output_dir / "censoring_by_condition.csv", index=False)
    print("\nCensoring by colour and speed:")
    print(censor_table.to_string(index=False))

    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    stats_comparison = None
    lrt = None

    if metadata["n_censored"] == 0:
        # With no censoring, the original MixedLM analysis remains valid.
        _, stats_comparison, lrt = run_statsmodels(
            dat,
            metadata["trial_order_adjusted"],
            args.output_dir,
        )
    else:
        print("\n" + "=" * 78)
        print("STATSMODELS NOTE")
        print("=" * 78)
        print(
            "Right-censored trials are present, so the original statsmodels MixedLM "
            "analysis is being skipped. MixedLM cannot represent censoring and would "
            "incorrectly treat censoring times as observed slowing times. The PyMC "
            "hierarchical censored AFT model is the primary analysis."
        )

    models, idatas, loo_comparison, region_probs, effect_table = run_pymc(
        dat,
        metadata,
        args,
        args.output_dir,
    )

    write_readme(
        args.output_dir,
        metadata,
        stats_comparison,
        lrt,
        loo_comparison,
        region_probs,
        args,
    )

    print("\n" + "=" * 78)
    print("ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Results written to: {args.output_dir.resolve()}")
    print(
        "\nFor censored data, base the biological conclusion primarily on the "
        "Bayesian PSIS-LOO comparison and posterior latency-ratio estimates."
    )

    # Returning results makes the function convenient to call from notebooks,
    # scripts, and interactive Python sessions.
    return {
        "data": dat,
        "metadata": metadata,
        "censoring_by_condition": censor_table,
        "statsmodels_model_comparison": stats_comparison,
        "statsmodels_lrt": lrt,
        "pymc_models": models,
        "inference_data": idatas,
        "loo_comparison": loo_comparison,
        "hypothesis_region_probabilities": region_probs,
        "effect_table": effect_table,
        "output_dir": args.output_dir,
    }


if __name__ == "__main__":
    csv_file = "out_censored.csv"
    main(csv_file)