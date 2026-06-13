import numpy as np


OHLC_FEATURES = ("High", "Low", "Close")


def ohlc_feature_matrix(df, assets=None):
    """Return features in asset-major order: asset_1 H/L/C, asset_2 H/L/C, ..."""
    feature_frames = []
    for feature in OHLC_FEATURES:
        frame = df.xs(feature, level=1, axis=1)
        if assets is not None:
            frame = frame.loc[:, list(assets)]
        feature_frames.append(frame)
    return np.stack([frame.to_numpy(dtype=float) for frame in feature_frames], axis=2).reshape(len(df), -1)
