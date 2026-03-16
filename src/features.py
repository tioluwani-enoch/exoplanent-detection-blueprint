def compute_features(period, duration, depth_raw, flux_out, flux_in, stellar_radius):
    """
    Transform raw transit parameters into physically grounded ML features.
    Returns None if values fall outside physically valid ranges.
    """
    # Feature 1: Normalized transit depth
    norm_depth = (flux_out - flux_in) / flux_out

    # Feature 2: Duration-to-period ratio
    dur_period_ratio = duration / period

    # Feature 3: Planet-to-star radius ratio (from depth)
    radius_ratio = norm_depth ** 0.5

    # --- Physics sanity filters ---
    if not (0 < norm_depth < 0.1):
        return None  # Likely eclipsing binary
    if not (0.5 < period < 500):
        return None  # Outside observable/physical range
    if not (0.001 < dur_period_ratio < 0.1):
        return None  # Geometrically implausible

    return {
        "norm_depth":       norm_depth,
        "dur_period_ratio": dur_period_ratio,
        "radius_ratio":     radius_ratio,
    }