"""
Larmor Formula Test
===================
Classical EM predicts:
- Accelerating charge: P ∝ a² (Larmor)
- Rotating dipole: P ∝ ω⁴ (magnetic dipole) or ω⁶ (electric dipole)

For our spinning disk in corrugated vacuum:
- Each point has acceleration a = ω²r (centripetal)
- Point velocity v = ωr
- Coupling to field gradient: F ∝ (v·∇h)²

Let's measure P(ω) and fit the power law exponent.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import laplace

# =============================================================================
# FIELD AND DISK (simplified for speed)
# =============================================================================

class FastField:
    """Optimized field for parameter sweeps"""

    def __init__(self, N=100, L=8.0, A=0.3, wavelength=1.0, c=5.0):
        self.N, self.L, self.A, self.c = N, L, A, c
        self.dx = L / N
        self.k = 2 * np.pi / wavelength

        x = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(x, x)
        self.h0 = A * np.sin(self.k * self.X)

        self.phi = np.zeros((N, N))
        self.phi_dot = np.zeros((N, N))

    def grad(self, x, y):
        bg = np.array([self.A * self.k * np.cos(self.k * x), 0.0])
        i = int((x + self.L/2) / self.dx)
        j = int((y + self.L/2) / self.dx)
        i, j = max(1, min(self.N-2, i)), max(1, min(self.N-2, j))
        dyn = np.array([
            (self.phi[j, i+1] - self.phi[j, i-1]) / (2*self.dx),
            (self.phi[j+1, i] - self.phi[j-1, i]) / (2*self.dx)
        ])
        return bg + dyn

    def deposit(self, x, y, amount, sigma=0.15):
        r2 = (self.X - x)**2 + (self.Y - y)**2
        self.phi_dot += amount * np.exp(-r2 / (2*sigma**2))

    def step(self, dt):
        # Adaptive substeps for stability
        cfl = self.c * dt / self.dx
        n_sub = max(1, int(np.ceil(cfl / 0.4)))
        sub_dt = dt / n_sub

        for _ in range(n_sub):
            lap = laplace(self.phi) / (self.dx**2)
            self.phi_dot *= 0.999  # Small damping
            self.phi_dot += self.c**2 * lap * sub_dt
            self.phi += self.phi_dot * sub_dt

            # Boundary absorption
            edge = int(0.1 * self.N)
            self.phi_dot[:edge, :] *= 0.9
            self.phi_dot[-edge:, :] *= 0.9
            self.phi_dot[:, :edge] *= 0.9
            self.phi_dot[:, -edge:] *= 0.9


class FastDisk:
    """Optimized spinning disk"""

    def __init__(self, radius=1.0, omega=5.0, n_points=24, alpha=0.01):
        self.radius = radius
        self.omega = omega
        self.theta = 0.0
        self.alpha = alpha
        self.n_points = n_points
        self.I = 0.5 * radius**2  # Unit mass

        self.power_samples = []

    def step(self, field, dt):
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        total_torque = 0.0
        total_power = 0.0

        for angle in angles:
            px = self.radius * np.cos(angle)
            py = self.radius * np.sin(angle)

            if abs(px) > field.L/2 - 0.5 or abs(py) > field.L/2 - 0.5:
                continue

            vx = -self.omega * self.radius * np.sin(angle)
            vy = self.omega * self.radius * np.cos(angle)

            dh = field.grad(px, py)
            v_dot_dh = vx * dh[0] + vy * dh[1]

            Fx = -2 * self.alpha * v_dot_dh * dh[0]
            Fy = -2 * self.alpha * v_dot_dh * dh[1]

            # Clamp
            Fx = np.clip(Fx, -50, 50)
            Fy = np.clip(Fy, -50, 50)

            torque = px * Fy - py * Fx
            total_torque += torque

            power = -(Fx * vx + Fy * vy)
            if 0 < power < 500:
                field.deposit(px, py, power * dt * 0.05, sigma=0.12)
                total_power += power

        # Update rotation
        self.omega += np.clip(total_torque / self.I, -50, 50) * dt
        self.theta += self.omega * dt

        self.power_samples.append(total_power)
        return total_power


# =============================================================================
# POWER LAW TEST
# =============================================================================

def measure_power_vs_omega():
    """
    Measure average radiated power as function of initial angular velocity.
    """
    print("=" * 60)
    print("LARMOR FORMULA TEST: P(ω) power law")
    print("=" * 60)

    # Range of angular velocities to test
    omegas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

    # Fixed parameters
    radius = 1.0
    coupling = 0.008

    measured_powers = []

    for omega in omegas:
        print(f"\nTesting ω = {omega}...")

        field = FastField(N=80, L=8.0, A=0.3, wavelength=1.0, c=5.0)
        disk = FastDisk(radius=radius, omega=omega, n_points=32, alpha=coupling)

        dt = 0.01
        warmup_steps = 50  # Let system settle
        measure_steps = 200

        # Warmup
        for _ in range(warmup_steps):
            disk.step(field, dt)
            field.step(dt)

        # Measure
        powers = []
        for _ in range(measure_steps):
            p = disk.step(field, dt)
            field.step(dt)
            powers.append(p)

        avg_power = np.mean(powers)
        measured_powers.append(avg_power)
        print(f"  Average power: {avg_power:.4f}")

    measured_powers = np.array(measured_powers)

    # Fit power law: P = A * ω^n
    def power_law(omega, A, n):
        return A * omega**n

    # Filter out any zeros or negatives
    valid = measured_powers > 0
    if np.sum(valid) >= 3:
        popt, pcov = curve_fit(power_law, omegas[valid], measured_powers[valid],
                               p0=[0.01, 2.0], maxfev=5000)
        A_fit, n_fit = popt
        n_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0
    else:
        A_fit, n_fit, n_err = 0.01, 2.0, 0

    print(f"\n" + "=" * 60)
    print(f"RESULT: P ∝ ω^{n_fit:.2f} ± {n_err:.2f}")
    print("=" * 60)
    print(f"\nFor comparison:")
    print(f"  Larmor (accelerating charge): P ∝ a² ∝ ω⁴")
    print(f"  Magnetic dipole radiation:    P ∝ ω⁴")
    print(f"  Electric dipole radiation:    P ∝ ω⁶")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear plot
    ax = axes[0]
    ax.plot(omegas, measured_powers, 'bo-', markersize=10, linewidth=2, label='Measured')
    omega_fit = np.linspace(omegas.min(), omegas.max(), 100)
    ax.plot(omega_fit, power_law(omega_fit, A_fit, n_fit), 'r--', linewidth=2,
            label=f'Fit: P ∝ ω^{n_fit:.2f}')
    ax.set_xlabel('Angular velocity ω', fontsize=12)
    ax.set_ylabel('Radiated Power P', fontsize=12)
    ax.set_title('Power vs Angular Velocity', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Log-log plot (should be linear if power law)
    ax = axes[1]
    ax.loglog(omegas[valid], measured_powers[valid], 'bo', markersize=10, label='Measured')
    ax.loglog(omega_fit, power_law(omega_fit, A_fit, n_fit), 'r--', linewidth=2,
              label=f'Slope = {n_fit:.2f}')

    # Reference lines
    ax.loglog(omega_fit, 0.001 * omega_fit**2, 'g:', alpha=0.5, label='∝ ω²')
    ax.loglog(omega_fit, 0.0001 * omega_fit**4, 'm:', alpha=0.5, label='∝ ω⁴')
    ax.loglog(omega_fit, 0.00001 * omega_fit**6, 'c:', alpha=0.5, label='∝ ω⁶')

    ax.set_xlabel('log(ω)', fontsize=12)
    ax.set_ylabel('log(P)', fontsize=12)
    ax.set_title(f'Log-Log Plot: Exponent = {n_fit:.2f} ± {n_err:.2f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('larmor_test.png', dpi=150)
    plt.show()

    return n_fit, n_err


def measure_power_vs_radius():
    """
    Measure how power scales with disk radius at fixed ω.
    For Larmor: P ∝ a² = (ω²r)² = ω⁴r²
    """
    print("\n" + "=" * 60)
    print("RADIUS SCALING TEST: P(r) at fixed ω")
    print("=" * 60)

    radii = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    omega = 5.0
    coupling = 0.008

    measured_powers = []

    for radius in radii:
        print(f"\nTesting r = {radius}...")

        field = FastField(N=80, L=10.0, A=0.3, wavelength=1.0, c=5.0)
        disk = FastDisk(radius=radius, omega=omega,
                       n_points=max(16, int(24*radius)), alpha=coupling)

        dt = 0.01
        warmup_steps = 50
        measure_steps = 200

        for _ in range(warmup_steps):
            disk.step(field, dt)
            field.step(dt)

        powers = []
        for _ in range(measure_steps):
            p = disk.step(field, dt)
            field.step(dt)
            powers.append(p)

        avg_power = np.mean(powers)
        measured_powers.append(avg_power)
        print(f"  Average power: {avg_power:.4f}")

    measured_powers = np.array(measured_powers)

    # Fit power law
    def power_law(r, A, n):
        return A * r**n

    valid = measured_powers > 0
    if np.sum(valid) >= 3:
        popt, pcov = curve_fit(power_law, radii[valid], measured_powers[valid],
                               p0=[0.1, 2.0], maxfev=5000)
        A_fit, n_fit = popt
        n_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0
    else:
        A_fit, n_fit, n_err = 0.1, 2.0, 0

    print(f"\n" + "=" * 60)
    print(f"RESULT: P ∝ r^{n_fit:.2f} ± {n_err:.2f}")
    print("=" * 60)
    print(f"\nFor Larmor formula: P ∝ r² (since a = ω²r)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(radii[valid], measured_powers[valid], 'bo', markersize=10, label='Measured')
    r_fit = np.linspace(radii.min(), radii.max(), 100)
    ax.loglog(r_fit, power_law(r_fit, A_fit, n_fit), 'r--', linewidth=2,
              label=f'Fit: P ∝ r^{n_fit:.2f}')
    ax.loglog(r_fit, 0.1 * r_fit**2, 'g:', alpha=0.5, label='∝ r² (Larmor)')
    ax.loglog(r_fit, 0.05 * r_fit**4, 'm:', alpha=0.5, label='∝ r⁴')

    ax.set_xlabel('Radius r', fontsize=12)
    ax.set_ylabel('Radiated Power P', fontsize=12)
    ax.set_title(f'Power vs Radius (ω={omega} fixed)\nExponent = {n_fit:.2f}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('radius_scaling.png', dpi=150)
    plt.show()

    return n_fit, n_err


def combined_scaling():
    """
    Test combined scaling P(ω, r) to see if it matches Larmor: P ∝ ω⁴r²
    """
    print("\n" + "=" * 60)
    print("COMBINED SCALING: P(ω, r)")
    print("=" * 60)

    # Test points
    test_cases = [
        (2.0, 0.5), (2.0, 1.0), (2.0, 1.5),
        (4.0, 0.5), (4.0, 1.0), (4.0, 1.5),
        (6.0, 0.5), (6.0, 1.0), (6.0, 1.5),
    ]

    results = []
    coupling = 0.008

    for omega, radius in test_cases:
        print(f"  ω={omega}, r={radius}...", end=" ")

        field = FastField(N=80, L=10.0, A=0.3, wavelength=1.0, c=5.0)
        disk = FastDisk(radius=radius, omega=omega, n_points=32, alpha=coupling)

        dt = 0.01
        for _ in range(50):
            disk.step(field, dt)
            field.step(dt)

        powers = []
        for _ in range(150):
            p = disk.step(field, dt)
            field.step(dt)
            powers.append(p)

        avg_power = np.mean(powers)
        results.append((omega, radius, avg_power))
        print(f"P = {avg_power:.4f}")

    # Fit: P = A * ω^n1 * r^n2
    from scipy.optimize import minimize

    omegas = np.array([r[0] for r in results])
    radii = np.array([r[1] for r in results])
    powers = np.array([r[2] for r in results])

    def model(params):
        A, n1, n2 = params
        pred = A * omegas**n1 * radii**n2
        return np.sum((np.log(pred + 1e-10) - np.log(powers + 1e-10))**2)

    res = minimize(model, [0.01, 2, 2], method='Nelder-Mead')
    A_fit, n1_fit, n2_fit = res.x

    print(f"\n" + "=" * 60)
    print(f"COMBINED FIT: P ∝ ω^{n1_fit:.2f} × r^{n2_fit:.2f}")
    print("=" * 60)
    print(f"\nLarmor prediction: P ∝ ω⁴ × r²")
    print(f"Our model:         P ∝ ω^{n1_fit:.1f} × r^{n2_fit:.1f}")

    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Measured points
    ax.scatter(omegas, radii, powers, c='blue', s=100, label='Measured')

    # Fitted surface
    omega_grid = np.linspace(1.5, 7, 20)
    r_grid = np.linspace(0.4, 1.6, 20)
    O, R = np.meshgrid(omega_grid, r_grid)
    P_fit = A_fit * O**n1_fit * R**n2_fit
    ax.plot_surface(O, R, P_fit, alpha=0.3, color='red')

    ax.set_xlabel('ω')
    ax.set_ylabel('r')
    ax.set_zlabel('Power')
    ax.set_title(f'P ∝ ω^{n1_fit:.1f} × r^{n2_fit:.1f}')

    plt.tight_layout()
    plt.savefig('combined_scaling.png', dpi=150)
    plt.show()

    return n1_fit, n2_fit


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING LARMOR-LIKE SCALING IN CORRUGATED VACUUM")
    print("=" * 60)
    print("""
Classical electromagnetic predictions:
- Point charge acceleration: P = (q²/6πε₀c³) × a²
- For circular motion: a = ω²r, so P ∝ ω⁴r²
- For rotating dipole: P ∝ ω⁴ (magnetic) or ω⁶ (electric)

Let's see what our corrugated vacuum model gives...
""")

    n_omega, err_omega = measure_power_vs_omega()
    n_radius, err_radius = measure_power_vs_radius()
    n1, n2 = combined_scaling()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"""
Measured scaling laws:

  P ∝ ω^{n_omega:.2f} (expected: 4 for Larmor, 6 for dipole)
  P ∝ r^{n_radius:.2f} (expected: 2 for Larmor)

Combined: P ∝ ω^{n1:.1f} × r^{n2:.1f}
Larmor:   P ∝ ω⁴ × r²

Interpretation:
""")

    if 3.5 < n_omega < 4.5:
        print("  ✓ ω-scaling matches Larmor formula (ω⁴)!")
    elif 5.5 < n_omega < 6.5:
        print("  ✓ ω-scaling matches dipole radiation (ω⁶)!")
    elif 1.5 < n_omega < 2.5:
        print("  → ω² scaling suggests simple friction model")
    else:
        print(f"  → Novel scaling ω^{n_omega:.1f} - may indicate new physics regime")

    if 1.5 < n_radius < 2.5:
        print("  ✓ r-scaling matches Larmor formula (r²)!")
    else:
        print(f"  → r^{n_radius:.1f} scaling differs from Larmor")
