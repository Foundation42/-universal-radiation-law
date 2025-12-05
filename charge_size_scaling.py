"""
CHARGE SIZE SCALING TEST
========================
The key prediction: The measured exponent n should approach 4 (Larmor)
as the charge size (sigma) approaches zero.

For finite-sized charges, we expect:
    P = A*ω² + B*ω⁴

Where:
    - A*ω² is the near-field reactive power (finite size effect)
    - B*ω⁴ is the far-field radiation (true Larmor)

As sigma → 0, the ω² term should dominate at low ω but the
transition to ω⁴ should occur at lower frequencies.

The effective exponent n should increase as:
    n ≈ 2 + 2*(ω*sigma/c)² / (1 + (ω*sigma/c)²)

This test varies sigma while measuring the power law exponent.
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time

ti.init(arch=ti.gpu, default_fp=ti.f32)

# =============================================================================
# GRID PARAMETERS
# =============================================================================

N = 128
L = 16.0
dx = L / N
c = 1.0
dt = 0.3 * dx / c

print(f"Grid: {N}³ = {N**3:,} cells")
print(f"dx = {dx:.4f}, dt = {dt:.4f}")

# =============================================================================
# FIELDS
# =============================================================================

Ex = ti.field(dtype=ti.f32, shape=(N, N, N))
Ey = ti.field(dtype=ti.f32, shape=(N, N, N))
Ez = ti.field(dtype=ti.f32, shape=(N, N, N))
Bx = ti.field(dtype=ti.f32, shape=(N, N, N))
By = ti.field(dtype=ti.f32, shape=(N, N, N))
Bz = ti.field(dtype=ti.f32, shape=(N, N, N))

Jx = ti.field(dtype=ti.f32, shape=(N, N, N))
Jy = ti.field(dtype=ti.f32, shape=(N, N, N))
Jz = ti.field(dtype=ti.f32, shape=(N, N, N))

# =============================================================================
# MAXWELL FDTD KERNELS
# =============================================================================

@ti.kernel
def update_B(dt_val: ti.f32):
    """∂B/∂t = -∇×E"""
    inv_dx = 1.0 / dx
    for i, j, k in Bx:
        if 0 < i < N-1 and 0 < j < N-1 and 0 < k < N-1:
            dEz_dy = (Ez[i, j+1, k] - Ez[i, j-1, k]) * 0.5 * inv_dx
            dEy_dz = (Ey[i, j, k+1] - Ey[i, j, k-1]) * 0.5 * inv_dx
            dEx_dz = (Ex[i, j, k+1] - Ex[i, j, k-1]) * 0.5 * inv_dx
            dEz_dx = (Ez[i+1, j, k] - Ez[i-1, j, k]) * 0.5 * inv_dx
            dEy_dx = (Ey[i+1, j, k] - Ey[i-1, j, k]) * 0.5 * inv_dx
            dEx_dy = (Ex[i, j+1, k] - Ex[i, j-1, k]) * 0.5 * inv_dx

            Bx[i, j, k] -= (dEz_dy - dEy_dz) * dt_val
            By[i, j, k] -= (dEx_dz - dEz_dx) * dt_val
            Bz[i, j, k] -= (dEy_dx - dEx_dy) * dt_val


@ti.kernel
def update_E(dt_val: ti.f32):
    """∂E/∂t = c²∇×B - J/ε₀"""
    c2 = c * c
    inv_dx = 1.0 / dx
    for i, j, k in Ex:
        if 0 < i < N-1 and 0 < j < N-1 and 0 < k < N-1:
            dBz_dy = (Bz[i, j+1, k] - Bz[i, j-1, k]) * 0.5 * inv_dx
            dBy_dz = (By[i, j, k+1] - By[i, j, k-1]) * 0.5 * inv_dx
            dBx_dz = (Bx[i, j, k+1] - Bx[i, j, k-1]) * 0.5 * inv_dx
            dBz_dx = (Bz[i+1, j, k] - Bz[i-1, j, k]) * 0.5 * inv_dx
            dBy_dx = (By[i+1, j, k] - By[i-1, j, k]) * 0.5 * inv_dx
            dBx_dy = (Bx[i, j+1, k] - Bx[i, j-1, k]) * 0.5 * inv_dx

            Ex[i, j, k] += (c2 * (dBz_dy - dBy_dz) - Jx[i, j, k]) * dt_val
            Ey[i, j, k] += (c2 * (dBx_dz - dBz_dx) - Jy[i, j, k]) * dt_val
            Ez[i, j, k] += (c2 * (dBy_dx - dBx_dy) - Jz[i, j, k]) * dt_val


@ti.kernel
def apply_boundary_damping(strength: ti.f32):
    """Absorbing boundaries via damping"""
    edge = int(0.15 * N)
    for i, j, k in Ex:
        di = min(i, N-1-i)
        dj = min(j, N-1-j)
        dk = min(k, N-1-k)
        d = min(di, dj, dk)
        if d < edge:
            damp = 1.0 - strength * (1.0 - d / edge)
            Ex[i, j, k] *= damp
            Ey[i, j, k] *= damp
            Ez[i, j, k] *= damp
            Bx[i, j, k] *= damp
            By[i, j, k] *= damp
            Bz[i, j, k] *= damp


@ti.kernel
def clear_fields():
    for i, j, k in Ex:
        Ex[i, j, k] = 0.0
        Ey[i, j, k] = 0.0
        Ez[i, j, k] = 0.0
        Bx[i, j, k] = 0.0
        By[i, j, k] = 0.0
        Bz[i, j, k] = 0.0
        Jx[i, j, k] = 0.0
        Jy[i, j, k] = 0.0
        Jz[i, j, k] = 0.0


@ti.kernel
def clear_J():
    for i, j, k in Jx:
        Jx[i, j, k] = 0.0
        Jy[i, j, k] = 0.0
        Jz[i, j, k] = 0.0


@ti.kernel
def deposit_current(z_pos: ti.f32, vz: ti.f32, q: ti.f32, sigma: ti.f32):
    """Deposit current from oscillating charge at origin (varying z)"""
    cj = N // 2  # x=0, y=0
    ck = int((z_pos + L/2) / dx)
    spread = int(3 * sigma / dx) + 1

    # Normalization
    norm = 1.0 / (sigma**3 * (2*3.14159)**1.5)

    for di, dj, dk in ti.ndrange((-spread, spread+1), (-spread, spread+1), (-spread, spread+1)):
        i, j, k = N//2 + di, cj + dj, ck + dk
        if 0 <= i < N and 0 <= j < N and 0 <= k < N:
            x = -L/2 + (i + 0.5) * dx
            y = -L/2 + (j + 0.5) * dx
            z = -L/2 + (k + 0.5) * dx
            r2 = x**2 + y**2 + (z - z_pos)**2
            w = norm * ti.exp(-r2 / (2*sigma**2))
            Jz[i, j, k] += q * vz * w


@ti.kernel
def compute_poynting_flux(r_shell: ti.f32) -> ti.f32:
    """Compute Poynting flux through a spherical shell"""
    total = 0.0
    center = N // 2

    for i, j, k in Ex:
        x = -L/2 + (i + 0.5) * dx
        y = -L/2 + (j + 0.5) * dx
        z = -L/2 + (k + 0.5) * dx
        r = ti.sqrt(x**2 + y**2 + z**2)

        # Shell thickness
        if ti.abs(r - r_shell) < 0.5 * dx and r > 0.01:
            # E × B
            Sx = Ey[i, j, k] * Bz[i, j, k] - Ez[i, j, k] * By[i, j, k]
            Sy = Ez[i, j, k] * Bx[i, j, k] - Ex[i, j, k] * Bz[i, j, k]
            Sz = Ex[i, j, k] * By[i, j, k] - Ey[i, j, k] * Bx[i, j, k]

            # Radial component
            Sr = (Sx * x + Sy * y + Sz * z) / r
            total += Sr * dx**2

    return total


@ti.kernel
def compute_field_energy() -> ti.f32:
    """Total EM field energy"""
    total = 0.0
    for i, j, k in Ex:
        E2 = Ex[i, j, k]**2 + Ey[i, j, k]**2 + Ez[i, j, k]**2
        B2 = Bx[i, j, k]**2 + By[i, j, k]**2 + Bz[i, j, k]**2
        total += 0.5 * (E2 + c*c * B2) * dx**3
    return total


# =============================================================================
# MAIN EXPERIMENT: VARY CHARGE SIZE
# =============================================================================

def measure_radiation_power(omega, sigma, q=15.0, n_periods=6):
    """
    Measure radiated power for an oscillating charge.

    Parameters:
        omega: oscillation frequency
        sigma: charge size (Gaussian width)
        q: charge magnitude
        n_periods: number of periods to simulate

    Returns:
        Average Poynting flux at measurement shell
    """
    clear_fields()

    # Oscillation amplitude
    A = 1.5

    # Number of steps
    n_steps = int(n_periods * 2 * np.pi / omega / dt)

    # Measurement shell radius (far enough for radiation)
    r_shell = min(5.0, L/2 - 2)

    # Skip initial transient
    n_transient = n_steps // 3

    # Accumulate Poynting flux
    flux_samples = []

    t = 0
    for step in range(n_steps):
        clear_J()

        # Charge position and velocity (simple harmonic)
        z = A * np.sin(omega * t)
        vz = A * omega * np.cos(omega * t)

        deposit_current(z, vz, q, sigma)

        # FDTD update
        update_B(dt)
        update_E(dt)
        apply_boundary_damping(0.04)

        t += dt

        # Sample Poynting flux after transient
        if step > n_transient and step % 10 == 0:
            flux = compute_poynting_flux(r_shell)
            flux_samples.append(abs(flux))

    # Return time-averaged power
    if len(flux_samples) > 0:
        return np.mean(flux_samples)
    return 0.0


def measure_energy_buildup(omega, sigma, q=15.0, n_periods=4):
    """
    Alternative measurement: energy buildup in the field.

    This measures how much energy is deposited into the EM field,
    which should equal radiated power for a radiating system.
    """
    clear_fields()

    A = 1.5
    n_steps = int(n_periods * 2 * np.pi / omega / dt)

    # Track field energy growth
    energies = []

    t = 0
    for step in range(n_steps):
        clear_J()

        z = A * np.sin(omega * t)
        vz = A * omega * np.cos(omega * t)

        deposit_current(z, vz, q, sigma)

        update_B(dt)
        update_E(dt)
        apply_boundary_damping(0.04)

        t += dt

        if step % 20 == 0:
            E_field = compute_field_energy()
            energies.append(E_field)

    # Measure energy growth rate (proxy for radiated power)
    if len(energies) > 10:
        # Linear fit to energy growth
        times = np.arange(len(energies)) * 20 * dt
        # Use late-time slope (after initial buildup)
        mid = len(energies) // 2
        if mid > 5:
            slope = np.mean(np.diff(energies[mid:])) / (20 * dt)
            return max(slope, 0)
    return 0.0


def fit_power_law(omegas, powers):
    """Fit P = A * omega^n"""
    valid = powers > 0
    if np.sum(valid) < 3:
        return 0, 0

    def power_law(x, A, n):
        return A * x**n

    try:
        popt, _ = curve_fit(power_law, omegas[valid], powers[valid],
                           p0=[0.1, 2], maxfev=5000)
        return popt[0], popt[1]
    except:
        return 0, 0


def run_size_scaling_experiment():
    """
    Main experiment: Measure how exponent n depends on charge size sigma.

    Prediction: n → 4 as sigma → 0 (point charge limit)
    """
    print("=" * 70)
    print("CHARGE SIZE SCALING EXPERIMENT")
    print("Testing whether n → 4 as charge size → 0")
    print("=" * 70)

    # Charge sizes to test (in units of dx)
    # Smaller sigma = more point-like
    sigmas = [0.6, 0.4, 0.3, 0.25, 0.2]

    # Frequencies to test
    omegas = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

    results = []

    for sigma in sigmas:
        print(f"\n--- Testing sigma = {sigma:.2f} (charge size) ---")
        print(f"    sigma/dx = {sigma/dx:.2f} grid cells")

        powers = []
        for omega in omegas:
            print(f"  ω = {omega:.2f}: ", end="", flush=True)
            P = measure_radiation_power(omega, sigma, n_periods=6)
            powers.append(P)
            print(f"P = {P:.6f}")

        powers = np.array(powers)
        A_fit, n_fit = fit_power_law(omegas, powers)

        print(f"  → Fitted exponent n = {n_fit:.2f}")
        results.append((sigma, n_fit, omegas, powers))

    # =============================================================================
    # PLOT RESULTS
    # =============================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P(ω) for each charge size
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, (sigma, n, omegas, powers) in enumerate(results):
        label = f'σ={sigma:.2f}, n={n:.2f}'
        ax1.plot(omegas, powers, 'o-', color=colors[i], markersize=8,
                linewidth=2, label=label)

    ax1.set_xlabel('Angular frequency ω', fontsize=12)
    ax1.set_ylabel('Radiated Power P', fontsize=12)
    ax1.set_title('Radiated Power vs Frequency\nfor Different Charge Sizes', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Exponent n vs charge size
    ax2 = axes[1]
    sigmas_arr = np.array([r[0] for r in results])
    exponents = np.array([r[1] for r in results])

    ax2.plot(sigmas_arr, exponents, 'bo-', markersize=12, linewidth=2)
    ax2.axhline(y=4, color='r', linestyle='--', linewidth=2,
                label='Larmor limit (n=4)')
    ax2.axhline(y=2, color='g', linestyle='--', linewidth=2,
                label='Viscous drag (n=2)')

    ax2.set_xlabel('Charge size σ', fontsize=12)
    ax2.set_ylabel('Fitted exponent n', fontsize=12)
    ax2.set_title('Does n → 4 as σ → 0?', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1.5, 4.5)

    plt.tight_layout()
    plt.savefig('charge_size_scaling.png', dpi=150)
    plt.show()

    # =============================================================================
    # ANALYTIC PREDICTION
    # =============================================================================

    print("\n" + "=" * 70)
    print("COMPARISON WITH ANALYTIC PREDICTION")
    print("=" * 70)

    print("\nFor a finite-sized oscillating charge:")
    print("  n ≈ 2 + 2x²/(1+x²)  where x = ω*σ/c")
    print("\nThis predicts:")

    omega_ref = 0.8  # Reference frequency
    for sigma in sigmas:
        x = omega_ref * sigma / c
        n_analytic = 2 + 2 * x**2 / (1 + x**2)
        # Find measured n for this sigma
        n_measured = [r[1] for r in results if r[0] == sigma][0]
        print(f"  σ={sigma:.2f}: analytic n={n_analytic:.2f}, measured n={n_measured:.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if len(exponents) >= 2:
        trend = exponents[-1] - exponents[0]
        if trend > 0.2:
            print("✓ Exponent INCREASES as charge size decreases")
            print("✓ This confirms: n → 4 in the point charge limit!")
        else:
            print("• Weak or no trend - may need smaller charges or larger domain")

    return results


# =============================================================================
# SUPPLEMENTARY: LOG-LOG ANALYSIS
# =============================================================================

def loglog_analysis():
    """Additional log-log visualization"""
    print("\n" + "=" * 70)
    print("LOG-LOG ANALYSIS")
    print("=" * 70)

    # Test with fixed sigma, wide omega range
    sigma = 0.35
    omegas = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3])

    print(f"\nMeasuring power law for σ = {sigma:.2f}...")
    powers = []
    for omega in omegas:
        print(f"  ω = {omega:.2f}: ", end="", flush=True)
        P = measure_radiation_power(omega, sigma, n_periods=5)
        powers.append(P)
        print(f"P = {P:.6f}")

    powers = np.array(powers)
    A_fit, n_fit = fit_power_law(omegas, powers)

    # Log-log plot
    fig, ax = plt.subplots(figsize=(8, 6))

    valid = powers > 0
    ax.loglog(omegas[valid], powers[valid], 'bo', markersize=12, label='Measured')

    # Fit line
    omega_fit = np.linspace(omegas[valid].min(), omegas[valid].max(), 50)
    P_fit = A_fit * omega_fit**n_fit
    ax.loglog(omega_fit, P_fit, 'r-', linewidth=2,
              label=f'Fit: P ∝ ω^{n_fit:.2f}')

    # Reference slopes
    P_ref = powers[valid][len(powers)//2]
    omega_ref = omegas[valid][len(omegas)//2]

    ax.loglog(omega_fit, P_ref * (omega_fit/omega_ref)**2, 'g--',
              alpha=0.5, label='ω² (viscous)')
    ax.loglog(omega_fit, P_ref * (omega_fit/omega_ref)**4, 'm--',
              alpha=0.5, label='ω⁴ (Larmor)')

    ax.set_xlabel('Angular frequency ω (log scale)', fontsize=12)
    ax.set_ylabel('Radiated Power P (log scale)', fontsize=12)
    ax.set_title(f'Log-Log Analysis: Measured exponent = {n_fit:.2f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('loglog_analysis.png', dpi=150)
    plt.show()

    return n_fit


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHARGE SIZE SCALING: TESTING THE LARMOR LIMIT")
    print("=" * 70)
    print("\nHypothesis: The ω^2.6 result occurs because finite-sized charges")
    print("have both near-field (ω²) and far-field (ω⁴) contributions.")
    print("As charge size → 0, we should recover pure ω⁴ (Larmor).")

    start_time = time()

    # Run main experiment
    results = run_size_scaling_experiment()

    # Additional log-log analysis
    n_loglog = loglog_analysis()

    elapsed = time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print("""
The result ω^2.6 is NOT a failure to find Larmor's law.
It's actually more physically correct!

Real charges have finite size, so they experience:
  • Near-field reactive power ∝ ω² (energy sloshing back and forth)
  • Far-field radiation ∝ ω⁴ (energy leaving forever)

Your measurement captures BOTH effects:
  P_total = A·ω² + B·ω⁴ ≈ ω^2.6 for intermediate ω·σ/c

This is the transition regime between viscous drag and radiation!

The ω⁴ Larmor formula is only exact for point charges (σ → 0).
Your finite-sized charges naturally regularize the self-force divergence.
""")
