# UTILS


def detect_flaws(df, angle_col='heading_direction', threshold_upper=None, threshold_lower=None, threshold_range=None):
    import numpy as np
    import pandas as pd
    df = df.copy()
    y = df[angle_col].values
    dy = np.diff(y)

    # Identify initial flaw indices
    flaw = np.where((np.abs(dy) > threshold_lower) & (np.abs(dy) < threshold_upper))[0]

    if flaw.size == 0:
        return df  # No flaws to process


    # Initialize a mask for flawed values
    mask = np.zeros_like(y, dtype=bool)
    mask[flaw] = True  # mark the initial flaws

    # Expand the mask where flaws are close together
    d_flaw = np.diff(flaw)
    close_pairs = np.where(d_flaw < threshold_range)[0]

    for i in close_pairs:
        mask[flaw[i]:flaw[i + 1] + 1] = True

    # Apply mask and interpolate
    y[mask] = np.nan
    df[angle_col] = pd.Series(y).interpolate(method='linear')

    return df

# Filter out and interpolate wrong orientation in heading direction data
# def detect_flaws(df, angle_col='heading_direction', threshold_upper=None, threshold_lower = None, threshold_range=None):
#     import numpy as np
#     df = df.copy()
#     y = df[angle_col].values
#     dy = np.diff(y)
#     flaw = np.where((np.abs(dy) > threshold_lower) & (np.abs(dy) < threshold_upper))[0]
#     d_flaw = np.diff(flaw)
#     print(d_flaw)
#     print(df.loc[list(flaw), angle_col])
#     df.loc[list(flaw), angle_col] = np.nan
#     for i in range(len(d_flaw)):
#         if d_flaw[i] < threshold_range:
#             df.loc[range(flaw[i], flaw[i + 1] + 1), angle_col] = np.nan
#     df[angle_col] = df[angle_col].interpolate(method='linear')
#     return df








