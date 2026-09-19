"""Analytical verification of the 2D shallow-water solver.

Run directly for a full report:   python tests/test_swe_benchmarks.py
Run under pytest for pass/fail:   pytest tests/test_swe_benchmarks.py
"""

import math

import numpy as np

from damburst.core.swe2d import SWE2D, G


def ritter(T=15.0, nx=1600, dx=0.5, h0=10.0, cfl=0.40, order=2):
    """Ritter (1892) dry-bed dam break; frictionless, horizontal bed."""
    ny = 3
    z = np.zeros((ny, nx))
    m = SWE2D(z, dx, dx, np.zeros((ny, nx)), open_edges=True, order=order)
    xc = (np.arange(nx) + 0.5) * dx
    x0 = nx * dx / 2
    m.h[:, xc < x0] = h0
    m.run(t_end=T, cfl=cfl, n_frames=0, dt_max=1.0)
    h = m.h[1, :]
    u = np.where(h > 1e-6, m.hu[1, :] / np.maximum(h, 1e-12), 0.0)
    x = xc - x0
    c0 = math.sqrt(G * h0)
    ha = np.where(x <= -c0 * T, h0,
                  np.where(x >= 2 * c0 * T, 0.0, (2 * c0 - x / T) ** 2 / (9 * G)))
    ua = np.where(x <= -c0 * T, 0.0,
                  np.where(x >= 2 * c0 * T, 0.0, 2.0 / 3.0 * (x / T + c0)))
    b = (x > -c0 * T) & (x < 2 * c0 * T * 0.98)
    return dict(l1h=float(np.abs(h[b] - ha[b]).mean()),
                l1u=float(np.abs(u[b] - ua[b]).mean()),
                front=float(xc[h > 1e-3].max() - x0),
                exact=2 * c0 * T)


def lake_at_rest(order=2, wetdry=True, T=600.0):
    """Still water over irregular bathymetry must stay exactly still."""
    rng = np.random.default_rng(0)
    ny, nx = 60, 120
    z = (6.0 * rng.random((ny, nx)) if wetdry else
         2.0 * rng.random((ny, nx)) + 3.0 * np.sin(np.linspace(0, 6, nx))[None, :])
    level = 3.0 if wetdry else 5.0
    m = SWE2D(z, 10.0, 10.0, np.full((ny, nx), 0.03), open_edges=False, order=order)
    m.set_still_water(level)
    r = m.run(t_end=T, cfl=0.45, n_frames=0)
    wet = m.h > 1e-2
    return dict(d_eta=float(np.abs(z[wet] + m.h[wet] - level).max()),
                v_max=float(r.v_max.max()))


def mass_closed(order=2):
    """A closed basin must conserve volume to round-off."""
    rng = np.random.default_rng(1)
    ny, nx = 80, 80
    z = 0.5 * rng.random((ny, nx))
    m = SWE2D(z, 5.0, 5.0, np.full((ny, nx), 0.025), open_edges=False, order=order)
    m.h[30:50, 30:50] = 4.0
    v0 = m.h.sum() * 25
    m.run(t_end=300.0, cfl=0.45, n_frames=0)
    return abs(m.h.sum() * 25 - v0) / v0


# --------------------------------------------------------------------------
# pytest assertions
# --------------------------------------------------------------------------

def test_lake_at_rest():
    for order in (1, 2):
        r = lake_at_rest(order=order)
        assert r["d_eta"] < 1e-8, (order, r)
        assert r["v_max"] < 1e-5, (order, r)


def test_mass_conservation():
    for order in (1, 2):
        assert mass_closed(order) < 1e-12


def test_ritter_accuracy():
    r = ritter(order=2)
    assert r["l1h"] < 0.02 * 10.0
    assert abs(r["front"] - r["exact"]) / r["exact"] < 0.08


if __name__ == "__main__":
    print("=== lake at rest (wet/dry, irregular bed) ===")
    for o in (1, 2):
        r = lake_at_rest(order=o)
        print(f"  order={o}: max|eta-level|={r['d_eta']:.3e} m   "
              f"max|u|={r['v_max']:.3e} m/s")

    print("=== closed-basin mass conservation ===")
    for o in (1, 2):
        print(f"  order={o}: rel.err={mass_closed(o):.3e}")

    print("=== Ritter dry-bed dam break, t=15 s, dx=0.5 m ===")
    for o in (1, 2):
        r = ritter(order=o)
        print(f"  order={o}: L1(h)={r['l1h']:.5f} m ({100 * r['l1h'] / 10:.2f}%)  "
              f"L1(u)={r['l1u']:.4f} m/s  front={r['front']:.1f} "
              f"exact={r['exact']:.1f} "
              f"err={100 * (r['front'] - r['exact']) / r['exact']:+.2f}%")

    print("=== grid convergence, Ritter t=15 s ===")
    for o in (1, 2):
        prev = None
        for nx, dx in ((400, 2.0), (800, 1.0), (1600, 0.5), (3200, 0.25)):
            r = ritter(15.0, nx, dx, order=o)
            rate = ("" if prev is None
                    else f"  rate={math.log(prev / r['l1h']) / math.log(2):.2f}")
            print(f"  order={o} dx={dx:5.2f} L1(h)={r['l1h']:.5f}{rate}"
                  f"   front err={100 * (r['front'] - r['exact']) / r['exact']:+6.2f}%")
            prev = r["l1h"]
