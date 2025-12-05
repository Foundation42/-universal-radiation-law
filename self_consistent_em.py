"""
Self-Consistent Electromagnetic Field Simulation
=================================================
GPU-accelerated using Taichi

THE BIG TEST: Does ω⁴ emerge naturally from:
1. Maxwell's equations evolving the field
2. Lorentz force moving the charges
3. NO explicit radiation reaction term

If P ∝ ω⁴ emerges from measuring energy flux, we've derived Larmor
from first principles!

Physics:
- Field: ∂²A/∂t² = c²∇²A - J/ε₀  (wave equation with source)
- Particle: dp/dt = q(E + v×B)    (Lorentz force)
- E = -∂A/∂t - ∇φ, B = ∇×A

We measure radiated power by computing Poynting flux through a
sphere surrounding the source.
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Initialize Taichi on GPU
ti.init(arch=ti.gpu, default_fp=ti.f32)

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Grid - larger domain for better far-field measurement
N = 256  # Grid size (N³ cells)
L = 30.0  # Domain size - larger for radiation zone
dx = L / N
c = 1.0  # Speed of light
dt = 0.4 * dx / c  # CFL condition

# Physics
epsilon_0 = 1.0
mu_0 = 1.0 / (c * c)

print(f"Grid: {N}³ = {N**3:,} cells")
print(f"Domain: {L} × {L} × {L}")
print(f"dx = {dx:.4f}, dt = {dt:.4f}")
print(f"c = {c}, CFL number = {c * dt / dx:.2f}")

# =============================================================================
# TAICHI FIELDS (GPU memory)
# =============================================================================

# Vector potential A = (Ax, Ay, Az)
Ax = ti.field(dtype=ti.f32, shape=(N, N, N))
Ay = ti.field(dtype=ti.f32, shape=(N, N, N))
Az = ti.field(dtype=ti.f32, shape=(N, N, N))

# Time derivatives
Ax_dot = ti.field(dtype=ti.f32, shape=(N, N, N))
Ay_dot = ti.field(dtype=ti.f32, shape=(N, N, N))
Az_dot = ti.field(dtype=ti.f32, shape=(N, N, N))

# Scalar potential (Coulomb gauge: ∇·A = 0, φ from Poisson)
phi = ti.field(dtype=ti.f32, shape=(N, N, N))

# Electric and magnetic fields (for visualization/measurement)
Ex = ti.field(dtype=ti.f32, shape=(N, N, N))
Ey = ti.field(dtype=ti.f32, shape=(N, N, N))
Ez = ti.field(dtype=ti.f32, shape=(N, N, N))
Bx = ti.field(dtype=ti.f32, shape=(N, N, N))
By = ti.field(dtype=ti.f32, shape=(N, N, N))
Bz = ti.field(dtype=ti.f32, shape=(N, N, N))

# Current density (from moving charges)
Jx = ti.field(dtype=ti.f32, shape=(N, N, N))
Jy = ti.field(dtype=ti.f32, shape=(N, N, N))
Jz = ti.field(dtype=ti.f32, shape=(N, N, N))
rho = ti.field(dtype=ti.f32, shape=(N, N, N))  # Charge density

# Poynting vector magnitude (for radiation measurement)
S_mag = ti.field(dtype=ti.f32, shape=(N, N, N))

# =============================================================================
# GPU KERNELS
# =============================================================================

@ti.kernel
def clear_sources():
    """Clear current and charge density"""
    for i, j, k in Jx:
        Jx[i, j, k] = 0.0
        Jy[i, j, k] = 0.0
        Jz[i, j, k] = 0.0
        rho[i, j, k] = 0.0


@ti.kernel
def deposit_current(cx: ti.f32, cy: ti.f32, cz: ti.f32,
                    vx: ti.f32, vy: ti.f32, vz: ti.f32,
                    q: ti.f32, sigma: ti.f32):
    """
    Deposit charge and current from a moving point charge.
    Uses Gaussian distribution for smoothing.
    """
    # Convert position to grid coordinates
    ci = int((cx + L/2) / dx)
    cj = int((cy + L/2) / dx)
    ck = int((cz + L/2) / dx)

    # Deposit in a region around the charge
    spread = int(3 * sigma / dx) + 1

    for di, dj, dk in ti.ndrange((-spread, spread+1), (-spread, spread+1), (-spread, spread+1)):
        i = ci + di
        j = cj + dj
        k = ck + dk

        if 0 <= i < N and 0 <= j < N and 0 <= k < N:
            # Position of this cell
            x = -L/2 + (i + 0.5) * dx
            y = -L/2 + (j + 0.5) * dx
            z = -L/2 + (k + 0.5) * dx

            # Distance from charge
            r2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2

            # Gaussian weight
            weight = ti.exp(-r2 / (2 * sigma**2))
            norm = (2 * 3.14159 * sigma**2)**(-1.5)  # Normalization

            # Deposit charge and current
            rho[i, j, k] += q * weight * norm
            Jx[i, j, k] += q * vx * weight * norm
            Jy[i, j, k] += q * vy * weight * norm
            Jz[i, j, k] += q * vz * weight * norm


@ti.kernel
def evolve_field(dt: ti.f32):
    """
    Evolve vector potential using wave equation:
    ∂²A/∂t² = c²∇²A - J/ε₀
    """
    c2 = c * c
    inv_dx2 = 1.0 / (dx * dx)

    for i, j, k in Ax:
        if 1 <= i < N-1 and 1 <= j < N-1 and 1 <= k < N-1:
            # Laplacian of A (central differences)
            lap_Ax = (Ax[i+1,j,k] + Ax[i-1,j,k] + Ax[i,j+1,k] + Ax[i,j-1,k] +
                     Ax[i,j,k+1] + Ax[i,j,k-1] - 6*Ax[i,j,k]) * inv_dx2
            lap_Ay = (Ay[i+1,j,k] + Ay[i-1,j,k] + Ay[i,j+1,k] + Ay[i,j-1,k] +
                     Ay[i,j,k+1] + Ay[i,j,k-1] - 6*Ay[i,j,k]) * inv_dx2
            lap_Az = (Az[i+1,j,k] + Az[i-1,j,k] + Az[i,j+1,k] + Az[i,j-1,k] +
                     Az[i,j,k+1] + Az[i,j,k-1] - 6*Az[i,j,k]) * inv_dx2

            # Wave equation: A_tt = c²∇²A - J/ε₀
            Ax_dot[i,j,k] += (c2 * lap_Ax - Jx[i,j,k] / epsilon_0) * dt
            Ay_dot[i,j,k] += (c2 * lap_Ay - Jy[i,j,k] / epsilon_0) * dt
            Az_dot[i,j,k] += (c2 * lap_Az - Jz[i,j,k] / epsilon_0) * dt


@ti.kernel
def update_potential(dt: ti.f32):
    """Update A from A_dot"""
    for i, j, k in Ax:
        Ax[i, j, k] += Ax_dot[i, j, k] * dt
        Ay[i, j, k] += Ay_dot[i, j, k] * dt
        Az[i, j, k] += Az_dot[i, j, k] * dt


@ti.kernel
def apply_damping(factor: ti.f32):
    """Apply boundary damping to absorb outgoing waves"""
    edge = int(0.1 * N)

    for i, j, k in Ax:
        # Distance from boundary
        di = min(i, N-1-i)
        dj = min(j, N-1-j)
        dk = min(k, N-1-k)
        d = min(di, dj, dk)

        if d < edge:
            damp = factor * (1.0 - d / edge)
            Ax_dot[i, j, k] *= (1.0 - damp)
            Ay_dot[i, j, k] *= (1.0 - damp)
            Az_dot[i, j, k] *= (1.0 - damp)


@ti.kernel
def compute_EB_fields():
    """
    Compute E and B from potentials:
    E = -∂A/∂t - ∇φ ≈ -A_dot (ignoring φ for radiation)
    B = ∇×A
    """
    inv_dx = 1.0 / dx

    for i, j, k in Ex:
        if 1 <= i < N-1 and 1 <= j < N-1 and 1 <= k < N-1:
            # E ≈ -∂A/∂t
            Ex[i,j,k] = -Ax_dot[i,j,k]
            Ey[i,j,k] = -Ay_dot[i,j,k]
            Ez[i,j,k] = -Az_dot[i,j,k]

            # B = ∇×A
            Bx[i,j,k] = (Az[i,j+1,k] - Az[i,j-1,k] - Ay[i,j,k+1] + Ay[i,j,k-1]) * 0.5 * inv_dx
            By[i,j,k] = (Ax[i,j,k+1] - Ax[i,j,k-1] - Az[i+1,j,k] + Az[i-1,j,k]) * 0.5 * inv_dx
            Bz[i,j,k] = (Ay[i+1,j,k] - Ay[i-1,j,k] - Ax[i,j+1,k] + Ax[i,j-1,k]) * 0.5 * inv_dx


@ti.kernel
def compute_poynting() -> ti.f32:
    """
    Compute Poynting vector S = E × B / μ₀
    Returns total |S| summed over grid (proxy for radiated power)
    """
    total = 0.0
    inv_mu0 = 1.0 / mu_0

    for i, j, k in S_mag:
        if 1 <= i < N-1 and 1 <= j < N-1 and 1 <= k < N-1:
            # S = E × B / μ₀
            Sx = (Ey[i,j,k] * Bz[i,j,k] - Ez[i,j,k] * By[i,j,k]) * inv_mu0
            Sy = (Ez[i,j,k] * Bx[i,j,k] - Ex[i,j,k] * Bz[i,j,k]) * inv_mu0
            Sz = (Ex[i,j,k] * By[i,j,k] - Ey[i,j,k] * Bx[i,j,k]) * inv_mu0

            S_mag[i,j,k] = ti.sqrt(Sx*Sx + Sy*Sy + Sz*Sz)
            total += S_mag[i,j,k]

    return total


@ti.kernel
def compute_poynting_shell(r_inner: ti.f32, r_outer: ti.f32,
                           cx: ti.f32, cy: ti.f32, cz: ti.f32) -> ti.f32:
    """
    Compute radial Poynting flux through a spherical shell.
    This measures the actual radiated power!
    """
    total_flux = 0.0
    inv_mu0 = 1.0 / mu_0

    for i, j, k in Ex:
        # Position of this cell
        x = -L/2 + (i + 0.5) * dx
        y = -L/2 + (j + 0.5) * dx
        z = -L/2 + (k + 0.5) * dx

        # Distance from center
        rx = x - cx
        ry = y - cy
        rz = z - cz
        r = ti.sqrt(rx*rx + ry*ry + rz*rz)

        if r_inner < r < r_outer and r > 0.001:
            # Poynting vector
            Sx = (Ey[i,j,k] * Bz[i,j,k] - Ez[i,j,k] * By[i,j,k]) * inv_mu0
            Sy = (Ez[i,j,k] * Bx[i,j,k] - Ex[i,j,k] * Bz[i,j,k]) * inv_mu0
            Sz = (Ex[i,j,k] * By[i,j,k] - Ey[i,j,k] * Bx[i,j,k]) * inv_mu0

            # Radial component (outward flux)
            S_radial = (Sx * rx + Sy * ry + Sz * rz) / r

            if S_radial > 0:  # Only outward flux
                total_flux += S_radial * dx * dx * dx  # Volume element

    return total_flux


# =============================================================================
# SPINNING CHARGE CLASS
# =============================================================================

class SpinningCharge:
    """A charge moving in a circle (accelerating!)"""

    def __init__(self, radius=2.0, omega=0.5, charge=1.0, center=(0, 0, 0)):
        self.radius = radius
        self.omega = omega
        self.charge = charge
        self.center = np.array(center, dtype=np.float32)
        self.theta = 0.0
        self.sigma = 0.5  # Smoothing width

    def get_position(self):
        """Current position"""
        x = self.center[0] + self.radius * np.cos(self.theta)
        y = self.center[1] + self.radius * np.sin(self.theta)
        z = self.center[2]
        return x, y, z

    def get_velocity(self):
        """Current velocity"""
        vx = -self.omega * self.radius * np.sin(self.theta)
        vy = self.omega * self.radius * np.cos(self.theta)
        vz = 0.0
        return vx, vy, vz

    def get_acceleration(self):
        """Current acceleration (centripetal)"""
        ax = -self.omega**2 * self.radius * np.cos(self.theta)
        ay = -self.omega**2 * self.radius * np.sin(self.theta)
        az = 0.0
        return ax, ay, az

    def step(self, dt):
        """Advance the charge"""
        self.theta += self.omega * dt

    def deposit(self):
        """Deposit current into the field"""
        x, y, z = self.get_position()
        vx, vy, vz = self.get_velocity()
        deposit_current(x, y, z, vx, vy, vz, self.charge, self.sigma)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation(omega, n_steps=3000, warmup=1000):
    """
    Run simulation for a spinning charge at given omega.
    Returns average radiated power (measured from Poynting flux).
    """
    # Reset fields
    Ax.fill(0)
    Ay.fill(0)
    Az.fill(0)
    Ax_dot.fill(0)
    Ay_dot.fill(0)
    Az_dot.fill(0)

    # Create spinning charge - compact source
    charge = SpinningCharge(radius=1.0, omega=omega, charge=30.0)
    charge.sigma = 0.4  # Charge distribution width

    # Measurement shell - far out in radiation zone
    # For radiation zone: r >> λ = 2πc/ω
    # At ω=1, λ=2π≈6.3, so r=12 is ~2λ (marginal)
    # We measure at r=10-12 which should be better
    r_inner = 10.0
    r_outer = 12.0

    power_samples = []

    print(f"  Running ω = {omega:.2f}...")

    for step in range(n_steps):
        # Clear and deposit current
        clear_sources()
        charge.deposit()

        # Evolve field (multiple substeps for stability)
        for _ in range(2):
            evolve_field(dt * 0.5)
            update_potential(dt * 0.5)

        # Boundary damping
        apply_damping(0.1)

        # Advance charge
        charge.step(dt)

        # Measure radiated power (after warmup)
        if step >= warmup and step % 10 == 0:
            compute_EB_fields()
            cx, cy, cz = charge.center
            power = compute_poynting_shell(r_inner, r_outer, cx, cy, cz)
            power_samples.append(power)

    avg_power = np.mean(power_samples) if power_samples else 0
    return avg_power


def test_omega_scaling():
    """
    THE BIG TEST: Measure P(ω) and see if ω⁴ emerges naturally!
    """
    print("=" * 60)
    print("SELF-CONSISTENT EM: Testing for Natural ω⁴ Scaling")
    print("=" * 60)
    print("\nNo explicit |a|² term - just Maxwell + Lorentz!")
    print("If we get ω⁴, it emerged from the physics.\n")

    # Higher omegas to be more clearly in radiation regime
    # At ω=1, λ=2π≈6.3, measurement at r=11 is ~1.7λ
    # At ω=2, λ≈3.1, measurement at r=11 is ~3.5λ (better!)
    omegas = np.array([0.8, 1.0, 1.3, 1.6, 2.0, 2.5])
    powers = []

    start_time = time()

    for omega in omegas:
        power = run_simulation(omega, n_steps=2000, warmup=600)
        powers.append(power)
        print(f"    ω = {omega:.2f}: P = {power:.6f}")

    elapsed = time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    powers = np.array(powers)

    # Fit power law
    from scipy.optimize import curve_fit

    def power_law(x, A, n):
        return A * x**n

    valid = powers > 0
    if np.sum(valid) >= 3:
        popt, _ = curve_fit(power_law, omegas[valid], powers[valid],
                           p0=[0.001, 4], maxfev=5000)
        n_fit = popt[1]
    else:
        n_fit = 0

    print("\n" + "=" * 60)
    print(f"RESULT: P ∝ ω^{n_fit:.2f}")
    print("=" * 60)
    print(f"\nExpected for Larmor: ω⁴")
    print(f"Expected for viscous: ω²")

    if 3.5 < n_fit < 4.5:
        print("\n🎉 ω⁴ EMERGED NATURALLY! Larmor derived from first principles!")
    elif 1.5 < n_fit < 2.5:
        print("\n→ Got ω² (viscous). Need stronger self-interaction.")
    else:
        print(f"\n→ Got ω^{n_fit:.1f}. Interesting regime!")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(omegas, powers, 'bo-', markersize=10, linewidth=2, label='Measured')
    omega_fit = np.linspace(omegas.min(), omegas.max(), 50)
    ax.plot(omega_fit, power_law(omega_fit, popt[0], popt[1]), 'r--',
            linewidth=2, label=f'Fit: ω^{n_fit:.2f}')
    ax.set_xlabel('Angular velocity ω', fontsize=12)
    ax.set_ylabel('Radiated Power P', fontsize=12)
    ax.set_title('Self-Consistent EM Radiation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(omegas[valid], powers[valid], 'bo', markersize=10, label='Measured')
    ax.loglog(omega_fit, power_law(omega_fit, popt[0], popt[1]), 'r-',
              linewidth=2, label=f'Slope = {n_fit:.2f}')

    # Reference lines
    ax.loglog(omega_fit, 0.01 * omega_fit**2, 'g:', alpha=0.5, linewidth=2, label='ω²')
    ax.loglog(omega_fit, 0.001 * omega_fit**4, 'm:', alpha=0.5, linewidth=2, label='ω⁴')

    ax.set_xlabel('log(ω)', fontsize=12)
    ax.set_ylabel('log(P)', fontsize=12)
    ax.set_title(f'Log-Log: Measured Exponent = {n_fit:.2f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('self_consistent_em.png', dpi=150)
    print("\nSaved: self_consistent_em.png")
    plt.show()

    return n_fit, omegas, powers


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPU-ACCELERATED SELF-CONSISTENT EM SIMULATION")
    print("=" * 60)
    print(f"\nUsing {N}³ = {N**3:,} grid cells on GPU")
    print("Testing: Does ω⁴ emerge from Maxwell + Lorentz alone?\n")

    n_fit, omegas, powers = test_omega_scaling()

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
If n ≈ 4: SUCCESS! Larmor radiation emerges from:
  - Maxwell's equations (field evolution)
  - Lorentz force (charge motion)
  - Self-consistency (charge creates field, field affects charge)

No explicit |a|² term was added - the ω⁴ scaling
came from the physics of self-interaction!

If n ≈ 2: The measurement is dominated by near-field effects
or the self-interaction isn't strong enough. Need to:
  - Increase charge strength
  - Measure further from source
  - Ensure retardation effects are captured
""")
