"""Core physics and geometry routines for the EOS BPM simulation.

The equations in this module are based on the supplied notebook
``EOS_BPM_Sim_3D_geom_2.ipynb``.  The refactor intentionally preserves the
existing numerical model while making the code importable and reusable.

All SI units are used internally unless explicitly noted in a parameter
name/docstring (for example, ``Ri`` and ``Ro`` are in millimetres to match
the original notebook interface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
from scipy.interpolate import interp1d

# Physical constants
C = 2.99792458e8  # m/s
HBAR = 1.055e-34  # J s
EPSILON_0 = 8.85e-12  # F/m

# GaP / material parameters from the original model
EPS_EL = 8.7
S0 = 1.8
FR0 = 10.98e12  # Hz
DEL0 = 0.02e12  # Hz
D_E = 1e-12  # m/V
C0 = -0.53


@dataclass(frozen=True)
class SimulationResult:
    """Outputs from :func:`simulate_eos_2b_finite`.

    Attributes
    ----------
    intensity:
        Simulated detector transmission fraction, dimensionless.
    gamma:
        Simulated phase shift in radians.
    time_map:
        Pulse-front-tilt time map in seconds.
    x, y:
        Crystal-plane coordinates in metres.
    """

    intensity: np.ndarray
    gamma: np.ndarray
    time_map: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __iter__(self):
        """Allow legacy tuple unpacking: ``intensity, gamma, t = result``."""
        yield self.intensity
        yield self.gamma
        yield self.time_map


def build_crystal_geometry(
    x: np.ndarray,
    y: np.ndarray,
    crystal_size: float = 10e-3,
    inner_gap_lr: float = 2e-3,
    inner_gap_tb: float = 2e-3,
) -> dict[str, np.ndarray]:
    """Build the four rectangular crystal masks.

    Parameters
    ----------
    x, y:
        One-dimensional crystal-plane coordinates in metres.
    crystal_size:
        Square crystal side length in metres.
    inner_gap_lr:
        Horizontal gap between left and right crystals in metres.
    inner_gap_tb:
        Vertical gap between top and bottom crystals in metres.

    Returns
    -------
    dict
        Boolean masks named ``top``, ``bottom``, ``left`` and ``right``.
    """

    X, Y = np.meshgrid(x, y, sparse=False)
    half = crystal_size / 2
    offset_x = half + inner_gap_lr / 2
    offset_y = half + inner_gap_tb / 2

    return {
        "top": (np.abs(X) <= half) & (np.abs(Y - offset_y) <= half),
        "bottom": (np.abs(X) <= half) & (np.abs(Y + offset_y) <= half),
        "left": (np.abs(X + offset_x) <= half) & (np.abs(Y) <= half),
        "right": (np.abs(X - offset_x) <= half) & (np.abs(Y) <= half),
    }


def plot_crystal_geometry(
    x: np.ndarray,
    y: np.ndarray,
    geometry: dict[str, np.ndarray],
    show_outlines: bool = True,
    ax=None,
):
    """Plot the four-crystal geometry and return the figure/axes."""

    from matplotlib.colors import BoundaryNorm, ListedColormap

    label = np.zeros_like(geometry["top"], dtype=int)
    label[geometry["top"]] = 1
    label[geometry["bottom"]] = 2
    label[geometry["left"]] = 3
    label[geometry["right"]] = 4

    overlap_count = sum(mask.astype(int) for mask in geometry.values())
    extent = [x.min() * 1e3, x.max() * 1e3, y.min() * 1e3, y.max() * 1e3]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    else:
        axes = np.asarray(ax).ravel()
        fig = axes[0].figure

    cmap = ListedColormap(["black", "tab:blue", "tab:green", "tab:orange", "tab:red"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    im0 = axes[0].imshow(
        label,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="equal",
    )
    axes[0].set_title("Crystal labels")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
    cbar0.ax.set_yticklabels(["none", "top", "bottom", "left", "right"])

    if show_outlines:
        for mask in geometry.values():
            axes[0].contour(
                mask.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.8,
                origin="lower",
                extent=extent,
            )

    im1 = axes[1].imshow(
        overlap_count,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[1].set_title("Crystal overlap count")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="count")

    if show_outlines:
        axes[1].contour(
            (overlap_count >= 2).astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.0,
            origin="lower",
            extent=extent,
        )

    return fig, axes


def plot_overlap_geometry(x, y, geometry, title="Crystal Geometry", ax=None):
    """Plot a simple geometry label map."""

    from matplotlib.colors import ListedColormap

    labels = np.zeros_like(next(iter(geometry.values())), dtype=np.int8)
    labels[geometry["top"]] = 1
    labels[geometry["bottom"]] = 2
    labels[geometry["left"]] = 3
    labels[geometry["right"]] = 4

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    else:
        fig = ax.figure

    cmap = ListedColormap(["black", "tab:blue", "tab:green", "tab:orange", "tab:red"])
    ax.imshow(
        labels,
        origin="lower",
        extent=[x.min() * 1e3, x.max() * 1e3, y.min() * 1e3, y.max() * 1e3],
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=4,
    )
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    return fig, ax


def FThz1d(t2d: np.ndarray, E2d: np.ndarray):
    """Return the center-row frequency axis, FFT field, and sample spacing."""

    ny, nx = t2d.shape
    cy = ny // 2
    t = t2d[cy, :]
    E = E2d[cy, :]
    dt = t[1] - t[0]

    E_f = fftshift(fft(ifftshift(E))) * dt
    f = fftshift(fftfreq(nx, d=dt))
    return f, E_f, dt


def ifft_response(f: np.ndarray, E_f: np.ndarray, dt: float) -> np.ndarray:
    """Inverse-transform a field produced by :func:`FThz1d`."""

    del f  # kept in the signature for backwards compatibility
    e_t = fftshift(ifft(ifftshift(E_f))) / dt
    return e_t.real


def recast_1d_to_2d(Rl: np.ndarray, E_1d: np.ndarray) -> np.ndarray:
    """Map a radial 1-D profile back onto a 2-D radius map."""

    ny, nx = Rl.shape
    cy = ny // 2
    cx = nx // 2

    r_uniform = Rl[cy, cx:]
    e_uniform = E_1d[cx:]
    r_unique, keep = np.unique(r_uniform, return_index=True)
    e_unique = e_uniform[keep]

    interpolator = interp1d(
        r_unique,
        e_unique,
        bounds_error=False,
        fill_value=0.0,
    )
    return interpolator(Rl)


# ---------------------------------------------------------------------------
# THz bunch fields
# ---------------------------------------------------------------------------


def E0_approx(r, Q, sig_t_thz):
    """Approximate peak radial THz field for a Gaussian bunch."""

    return (Q / (2 * np.pi * EPSILON_0 * r * C)) * (
        1 / (np.sqrt(2 * np.pi) * sig_t_thz)
    )


def Er_s(r, t, sig_t_thz, time_delay, Q):
    """Single-bunch radial field versus time."""

    return E0_approx(r, Q, sig_t_thz) * np.exp(
        -((t - time_delay) ** 2) / (2 * sig_t_thz**2)
    )


def E01_approx(r, Q1, sig_t1_thz):
    return E0_approx(r, Q1, sig_t1_thz)


def E02_approx(r, Q2, sig_t2_thz):
    return E0_approx(r, Q2, sig_t2_thz)


def Er_2(r, t, time_delay, delt12_thz, sig_t2_thz, sig_t1_thz, Q1, Q2):
    return Er1t(r, t, time_delay, sig_t1_thz, Q1) + Er2t(
        r, t, time_delay, delt12_thz, sig_t2_thz, Q2
    )


def Er2t(r2, t, time_delay, delt12_thz, sig_t2_thz, Q2):
    return E02_approx(r2, Q2, sig_t2_thz) * np.exp(
        -((t - time_delay - delt12_thz) ** 2) / (2 * sig_t2_thz**2)
    )


def Er1t(r1, t, time_delay, sig_t1_thz, Q1):
    return E01_approx(r1, Q1, sig_t1_thz) * np.exp(
        -((t - time_delay) ** 2) / (2 * sig_t1_thz**2)
    )


# ---------------------------------------------------------------------------
# GaP frequency response
# ---------------------------------------------------------------------------


def eps(f):
    return EPS_EL + (S0 * FR0**2) / (
        FR0**2 - f**2 - complex(0, DEL0) * f
    )


def n(f):
    return np.real(np.sqrt(eps(f)))


def k(f):
    return np.imag(np.sqrt(eps(f)))


def nopt(x):
    x_um = x * 1e6
    return np.sqrt(2.680 + (6.40 * x_um**2) / (x_um**2 - 0.0903279))


def dnopt(x):
    return 2529822.1281347 * (
        -1.0e-12 * x**3 / (x**2 - 9.03279e-14) ** 2
        + 1.0 * x / (1000000000000.0 * x**2 - 0.0903279)
    ) / (
        x**2 / (1000000000000.0 * x**2 - 0.0903279) + 4.1875e-13
    ) ** 0.5


def vg_opt(x):
    return (C / nopt(x)) * (1 + (x / nopt(x)) * dnopt(x))


def vg_opt_eff(x, alpha_l):
    return vg_opt(x) * np.cos(alpha_l)


def vg(f):
    droote = 5.06756393339777e-40 * (2 * f + complex(0, 20000000000.0)) / (
        (4.0090554886458e-26 + 1 / (-f**2 - complex(0, 20000000000.0) * f + 1.205604e26))
        ** 0.5
        * (-8.29459756271545e-27 * f**2 - complex(0, 1.65891951254309e-16) * f + 1) ** 2
    )
    return C / (n(f) + f * np.real(droote))


def vph(f):
    return C / n(f)


def r41(f):
    return D_E * (1 + ((C0 * FR0**2) / (FR0**2 - f**2 - complex(0, DEL0) * f)))


def Gd(f, d, vg_laser_eff):
    a = -complex(0, C) * (
        -1
        + np.exp(
            2
            * d
            * np.pi
            * f
            * (
                complex(0, 1) * ((-1 / vg_laser_eff) + (1 / vph(f)))
                - k(f) / C
            )
        )
    )
    b = vg_laser_eff * vph(f) / (
        2
        * np.pi
        * d
        * f
        * (-C * vph(f) + vg_laser_eff * (C + complex(0, 1) * k(f) * vph(f)))
    )
    return a * b


def G(f, d, vg_laser_eff, sig_eff_laser):
    """Finite laser-pulse response function."""

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        a = complex(0, C) * np.exp(
            -2
            * f
            * np.pi
            * (
                f * np.pi * sig_eff_laser**2
                + complex(0, 1 / vg_laser_eff) * d
                + d * k(f) / C
            )
        )
        b = np.exp(
            2 * d * np.pi * f * (complex(0, 1 / vg_laser_eff)) + k(f) / C
        ) - np.exp(complex(0, 2) * d * f * np.pi / vph(f))
        denominator = 2 * d * np.pi * f * (
            -C * vph(f) + vg_laser_eff * (C + complex(0, 1) * k(f) * vph(f))
        )
        geom = vg_laser_eff * vph(f) / denominator
        return a * b * geom


def Gd_interp(f, G_array):
    """Bridge non-finite response values, notably the undefined zero-frequency point."""

    G_array = np.asarray(G_array, dtype=complex).copy()
    bad = ~np.isfinite(G_array)
    if not np.any(bad):
        return G_array

    good = ~bad
    if not np.any(good):
        raise ValueError("Response function contains no finite frequency points.")

    G_array[bad] = np.interp(f[bad], f[good], G_array[good].real) + 1j * np.interp(
        f[bad], f[good], G_array[good].imag
    )
    return G_array


def Atr(f):
    return 2 / (1 + n(f) + complex(0, 1) * k(f))


def GEOd(f, G_array):
    return G_array * Atr(f) * r41(f)


def Aref(f):
    return (1 - n(f) - complex(0, 1) * k(f)) / (
        1 + n(f) + complex(0, 1) * k(f)
    )


def GEO_ref(f, d):
    return (Aref(f)) ** 2 * np.exp(
        complex(0, 1) * 2 * np.pi * f * n(f) * 2 * d / C
    ) * np.exp(-2 * np.pi * f * k(f) * 2 * d / C)


# ---------------------------------------------------------------------------
# Polarization / geometry response
# ---------------------------------------------------------------------------


def phi1(alpha):
    return 0.5 * np.arccos(
        np.sin(alpha) / np.sqrt(1 + 3 * np.cos(alpha) ** 2)
    )


def phi2(alpha):
    alpha_rot = alpha + np.pi / 2
    return 0.5 * np.arccos(
        np.sin(alpha_rot) / np.sqrt(1 + 3 * np.cos(alpha_rot) ** 2)
    )


# Backwards-compatible aliases for the notebook's unicode names.
φ1 = phi1
φ2 = phi2


def Gamma_1(alpha, laser_wavelength, dcry, total_field):
    Rlaser = (
        (np.pi / laser_wavelength)
        * nopt(laser_wavelength) ** 3
        * dcry
        * np.sqrt(1 + 3 * np.cos(alpha) ** 2)
    )
    return Rlaser * total_field


def Gamma_2(alpha, laser_wavelength, dcry, total_field):
    alpha_rot = alpha + np.pi / 2
    Rlaser = (
        (np.pi / laser_wavelength)
        * nopt(laser_wavelength) ** 3
        * dcry
        * np.sqrt(1 + 3 * np.cos(alpha_rot) ** 2)
    )
    return Rlaser * total_field


def time_tilt(Rlaser, ax_angle=10):
    """Pulse-front-tilt time across a radial distance ``Rlaser``."""

    theta1 = np.radians(ax_angle)
    ng = 1.4671
    n_g = 1.4533
    theta5 = np.pi / 2 - (
        theta1 * (n_g - 1)
        + np.arctan((1 - (theta1**2 * ng * (n_g - 1))) / (theta1 * (ng - 1)))
    )
    return Rlaser * (theta5 + theta1 * (n_g - 1)) / C



def _plot_simulation_result(result: SimulationResult):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    extent = [result.x.min() * 1e3, result.x.max() * 1e3, result.y.min() * 1e3, result.y.max() * 1e3]

    im0 = axes[0].imshow(
        result.intensity * 100,
        cmap="plasma",
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[0].set_title("Intensity")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="% Transmission")

    im1 = axes[1].imshow(
        result.gamma * 1e3,
        cmap="plasma",
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    axes[1].set_title(r"Phase Shift ($\Gamma$)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="mrad")

    return fig, axes



def simulate_eos_2b_finite(
    r01x,
    r01y,
    r02x,
    r02y,
    time_delay,
    *,
    Q1=1000e-12,
    Q2=800e-12,
    sig_1=15e-6 / C,
    sig_2=15e-6 / C,
    delt12=400e-15,
    dcry=100e-6,
    theta=5,
    ax_angle=10,
    pixelsx=2856,
    pixelsy=2856,
    Ri=5.2,
    Ro=13.2,
    laser=800e-9,
    sig_eff_laser=50e-15,
    crystal_size=10e-3,
    inner_gap_lr=2e-3,
    inner_gap_tb=2e-3,
    delay_topbottom=True,
    delay_amount=200e-15,
    plot=False,
):
    """Simulate a two-bunch EOS BPM response with four rectangular crystals.

    Parameters are kept close to the supplied notebook's latest implementation.
    ``Ri`` and ``Ro`` are in millimetres; all other lengths/times use SI units.

    Returns
    -------
    SimulationResult
        Contains intensity, phase shift, time map, and coordinate axes.
    """

    if pixelsx < 3 or pixelsy < 3:
        raise ValueError("pixelsx and pixelsy must both be at least 3.")
    if crystal_size <= 0:
        raise ValueError("crystal_size must be positive.")
    if Ro <= 0:
        raise ValueError("Ro must be positive.")
    if Ri < 0:
        raise ValueError("Ri must be non-negative.")

    crystx = Ro * 1e-3
    crysty = Ro * 1e-3
    x = np.linspace(-crystx, crystx, pixelsx)
    y = np.linspace(-crysty, crysty, pixelsy)
    X, Y = np.meshgrid(x, y, sparse=False)
    Rl = np.sqrt(X**2 + Y**2)

    Ric = Ri * 1e-3
    theta_rad = np.radians(theta)
    theta1 = np.radians(ax_angle)
    ng = 1.4671
    n_g = 1.4533
    theta5 = np.pi / 2 - (
        theta1 * (n_g - 1)
        + np.arctan((1 - (theta1**2 * ng * (n_g - 1))) / (theta1 * (ng - 1)))
    )

    Rlaser = Rl - Ric
    t = Rlaser * (theta5 + theta1 * (n_g - 1)) / C

    alphab1 = np.mod(np.arctan2(Y - r01y, X - r01x), 2 * np.pi)
    alphab2 = np.mod(np.arctan2(Y - r02y, X - r02x), 2 * np.pi)

    Rlo1 = np.sqrt((X - r01x) ** 2 + (Y - r01y) ** 2)
    Rlo2 = np.sqrt((X - r02x) ** 2 + (Y - r02y) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio1 = Rlo1 / Rl
        ratio2 = Rlo2 / Rl
    ratio1 = np.nan_to_num(ratio1, nan=0.0, posinf=0.0, neginf=0.0)
    ratio2 = np.nan_to_num(ratio2, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_total_fields(time_delay_eff):
        f1, E_rf1, dt = FThz1d(t, Er1t(Rl, t, time_delay_eff, sig_1, Q1))
        f2, E_rf2, _ = FThz1d(
            t, Er2t(Rl, t, time_delay_eff, delt12, sig_2, Q2)
        )

        response1 = Gd_interp(
            f1, G(f1, dcry, vg_opt_eff(laser, theta_rad), sig_eff_laser)
        )
        response2 = Gd_interp(
            f2, G(f2, dcry, vg_opt_eff(laser, theta_rad), sig_eff_laser)
        )

        field1 = E_rf1 * GEOd(f1, response1)
        field2 = E_rf2 * GEOd(f2, response2)
        reflected1 = field1 * GEO_ref(f1, dcry)
        reflected2 = field2 * GEO_ref(f2, dcry)

        trans_field1 = ifft_response(f1, field1, dt)
        trans_field2 = ifft_response(f2, field2, dt)
        ref_field1 = ifft_response(f1, reflected1, dt)
        ref_field2 = ifft_response(f2, reflected2, dt)

        with np.errstate(divide="ignore", invalid="ignore"):
            total_field1 = (
                recast_1d_to_2d(Rl, trans_field1)
                + recast_1d_to_2d(Rl, ref_field1)
            ) / ratio1
            total_field2 = (
                recast_1d_to_2d(Rl, trans_field2)
                + recast_1d_to_2d(Rl, ref_field2)
            ) / ratio2

        total_field1[t < 0] = 0
        total_field2[t < 0] = 0
        return (
            np.nan_to_num(total_field1, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(total_field2, nan=0.0, posinf=0.0, neginf=0.0),
        )

    total_field1_nom, total_field2_nom = compute_total_fields(time_delay)
    if delay_topbottom and delay_amount != 0:
        total_field1_del, total_field2_del = compute_total_fields(
            time_delay + delay_amount
        )
    else:
        total_field1_del, total_field2_del = total_field1_nom, total_field2_nom

    geometry = build_crystal_geometry(
        x,
        y,
        crystal_size=crystal_size,
        inner_gap_lr=inner_gap_lr,
        inner_gap_tb=inner_gap_tb,
    )
    m_union = np.logical_or.reduce(tuple(geometry.values()))

    gamma_map = np.zeros_like(Rl)
    intensity = np.zeros_like(Rl)

    def add_region(mask, phi_fn: Callable, gamma_fn: Callable, field1, field2):
        if not np.any(mask):
            return
        g1 = gamma_fn(alphab1[mask], laser, dcry, field1[mask])
        g2 = gamma_fn(alphab2[mask], laser, dcry, field2[mask])
        gamma_map[mask] += g1 + g2
        intensity[mask] += (
            np.sin(phi_fn(alphab1[mask])) ** 2 * np.sin(g1 / 2) ** 2
        )
        intensity[mask] += (
            np.sin(phi_fn(alphab2[mask])) ** 2 * np.sin(g2 / 2) ** 2
        )

    add_region(geometry["left"], phi1, Gamma_1, total_field1_nom, total_field2_nom)
    add_region(geometry["right"], phi1, Gamma_1, total_field1_nom, total_field2_nom)
    add_region(geometry["top"], phi2, Gamma_2, total_field1_del, total_field2_del)
    add_region(geometry["bottom"], phi2, Gamma_2, total_field1_del, total_field2_del)

    gamma_map[~m_union] = 0.0
    intensity[~m_union] = 0.0

    # Preserve the latest notebook implementation's final detector relation.
    intensity = np.sin(gamma_map / 2) ** 2

    result = SimulationResult(
        intensity=intensity,
        gamma=gamma_map,
        time_map=t,
        x=x,
        y=y,
    )

    if plot:
        fig, _ = _plot_simulation_result(result)
        fig.suptitle(f"Maximum intensity: {np.max(intensity) * 100:.3f}%")
        print(f"Maximum intensity: {np.max(intensity) * 100:.3f}%")
        plt.show()

    return result


# Legacy notebook-style API name.
EOS_sim_2b_finite = simulate_eos_2b_finite
