import numpy as np

from eos_bpm_sim import build_crystal_geometry, simulate_eos_2b_finite


def test_geometry_masks_have_expected_shape():
    x = np.linspace(-0.01, 0.01, 64)
    y = np.linspace(-0.01, 0.01, 48)
    geometry = build_crystal_geometry(x, y)

    assert set(geometry) == {"top", "bottom", "left", "right"}
    assert all(mask.shape == (48, 64) for mask in geometry.values())
    assert all(mask.dtype == bool for mask in geometry.values())


def test_small_simulation_runs():
    result = simulate_eos_2b_finite(
        r01x=0.0,
        r01y=0.0,
        r02x=0.0,
        r02y=0.0,
        time_delay=0.0,
        Q1=1e-12,
        Q2=0.0,
        pixelsx=32,
        pixelsy=24,
        plot=False,
    )

    assert result.intensity.shape == (24, 32)
    assert result.gamma.shape == (24, 32)
    assert result.time_map.shape == (24, 32)
    assert np.all(np.isfinite(result.intensity))
    assert np.all(np.isfinite(result.gamma))
    assert np.all(result.intensity >= 0)


def test_legacy_unpacking():
    result = simulate_eos_2b_finite(
        r01x=0.0,
        r01y=0.0,
        r02x=0.0,
        r02y=0.0,
        time_delay=0.0,
        pixelsx=12,
        pixelsy=12,
        plot=False,
    )
    intensity, gamma, time_map = result
    assert intensity is result.intensity
    assert gamma is result.gamma
    assert time_map is result.time_map
