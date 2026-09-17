"""Minimal example of the EOS BPM simulation."""

import matplotlib.pyplot as plt

from eos_bpm_sim import simulate_eos_2b_finite


result = simulate_eos_2b_finite(
    r01x=2e-3,
    r01y=0.0,
    r02x=0.0,
    r02y=0.0,
    time_delay=0.0,
    Q1=1.6e-9,
    Q2=0.0,
    pixelsx=512,
    pixelsy=512,
    plot=True,
)

print(f"Maximum transmission: {result.intensity.max() * 100:.3f}%")

plt.show()
