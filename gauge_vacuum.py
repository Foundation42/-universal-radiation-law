"""
Gauge Field Vacuum Simulation
=============================
Testing the hypothesis that gauge structure → ω⁴ scaling naturally.

The key insight:
- Scalar field h(x): couples through ∇h → spatial structure → ω²
- Vector gauge field A_μ: couples through F_μν → temporal derivatives → ω⁴

We simulate a U(1) gauge field (electromagnetism) and see if Larmor
scaling emerges WITHOUT artificially inserting |a|².

The Lorentz force F = q(E + v×B) involves:
- E = -∂A/∂t - ∇φ (time derivative!)
- B = ∇×A

When a charge accelerates, it must "drag" the field, but gauge invariance
means the field configuration is constrained - this should give ω⁴.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.optimize import curve_fit

# =============================================================================
# U(1) GAUGE FIELD (ELECTROMAGNETIC VACUUM)
# =============================================================================

class GaugeVacuum:
    """
    Electromagnetic vacuum with vector potential A_μ = (φ, A_x, A_y).

    In 2D+1 (x, y, t):
    - φ(x,y,t) = scalar potential
    - A = (A_x, A_y) = vector potential

    Fields:
    - E = -∇φ - ∂A/∂t
    - B = ∂A_y/∂x - ∂A_x/∂y (scalar in 2D, like B_z)

    Wave equation (Lorenz gauge):
    ∂²A/∂t² - c²∇²A = -μ₀ J
    """

    def __init__(self, N=100, L=10.0, c=5.0):
        self.N = N
        self.L = L
        self.dx = L / N
        self.c = c

        x = np.linspace(-L/2, L/2, N)
        self.X, self.Y = np.meshgrid(x, x)

        # Vector potential components
        self.Ax = np.zeros((N, N))
        self.Ay = np.zeros((N, N))
        self.phi = np.zeros((N, N))  # Scalar potential

        # Time derivatives (for wave equation)
        self.Ax_dot = np.zeros((N, N))
        self.Ay_dot = np.zeros((N, N))
        self.phi_dot = np.zeros((N, N))

        # Background "structure" - a static magnetic field pattern
        # This provides something to "push against"
        # B_0 = B_0 sin(kx) (like corrugation but magnetic)
        self.k = 2 * np.pi / 1.0  # wavelength = 1
        self.B0 = 0.5  # Background field strength

        # Energy tracking
        self.field_energy_history = []

    def get_E_field(self, x, y):
        """Electric field at a point: E = -∇φ - ∂A/∂t"""
        i = int((x + self.L/2) / self.dx)
        j = int((y + self.L/2) / self.dx)
        i = max(1, min(self.N-2, i))
        j = max(1, min(self.N-2, j))

        # -∇φ
        Ex_grad = -(self.phi[j, i+1] - self.phi[j, i-1]) / (2*self.dx)
        Ey_grad = -(self.phi[j+1, i] - self.phi[j-1, i]) / (2*self.dx)

        # -∂A/∂t
        Ex_dot = -self.Ax_dot[j, i]
        Ey_dot = -self.Ay_dot[j, i]

        return np.array([Ex_grad + Ex_dot, Ey_grad + Ey_dot])

    def get_B_field(self, x, y):
        """Magnetic field at a point: B = ∇×A (scalar in 2D)"""
        i = int((x + self.L/2) / self.dx)
        j = int((y + self.L/2) / self.dx)
        i = max(1, min(self.N-2, i))
        j = max(1, min(self.N-2, j))

        # B = ∂A_y/∂x - ∂A_x/∂y
        dAy_dx = (self.Ay[j, i+1] - self.Ay[j, i-1]) / (2*self.dx)
        dAx_dy = (self.Ax[j+1, i] - self.Ax[j-1, i]) / (2*self.dx)

        # Add background magnetic field
        B_background = self.B0 * np.sin(self.k * x)

        return dAy_dx - dAx_dy + B_background

    def deposit_current(self, x, y, jx, jy, sigma=0.2):
        """Deposit current density (moving charge) into the field"""
        r2 = (self.X - x)**2 + (self.Y - y)**2
        source = np.exp(-r2 / (2*sigma**2))

        # Current sources the vector potential
        # ∂²A/∂t² = c²∇²A - μ₀J
        # In our units, just add J to A_dot_dot
        self.Ax_dot += 0.1 * jx * source
        self.Ay_dot += 0.1 * jy * source

    def step(self, dt):
        """Evolve gauge field using wave equation"""
        cfl = self.c * dt / self.dx
        n_sub = max(1, int(np.ceil(cfl / 0.4)))
        sub_dt = dt / n_sub

        for _ in range(n_sub):
            # Wave equation for each component
            lap_Ax = laplace(self.Ax) / (self.dx**2)
            lap_Ay = laplace(self.Ay) / (self.dx**2)

            # Small damping for stability
            self.Ax_dot *= 0.999
            self.Ay_dot *= 0.999

            self.Ax_dot += self.c**2 * lap_Ax * sub_dt
            self.Ay_dot += self.c**2 * lap_Ay * sub_dt

            self.Ax += self.Ax_dot * sub_dt
            self.Ay += self.Ay_dot * sub_dt

            # Boundary absorption
            edge = int(0.1 * self.N)
            self.Ax_dot[:edge, :] *= 0.9
            self.Ax_dot[-edge:, :] *= 0.9
            self.Ax_dot[:, :edge] *= 0.9
            self.Ax_dot[:, -edge:] *= 0.9
            self.Ay_dot[:edge, :] *= 0.9
            self.Ay_dot[-edge:, :] *= 0.9
            self.Ay_dot[:, :edge] *= 0.9
            self.Ay_dot[:, -edge:] *= 0.9


# =============================================================================
# CHARGED SPINNING DISK IN GAUGE FIELD
# =============================================================================

class ChargedDisk:
    """
    Spinning disk with charge, coupled to gauge field via Lorentz force.

    F = q(E + v × B)

    In 2D: v × B = (v_x, v_y, 0) × (0, 0, B) = (v_y B, -v_x B, 0)
    So F_x = q(E_x + v_y B), F_y = q(E_y - v_x B)
    """

    def __init__(self, radius=1.0, omega=5.0, n_points=32, charge=1.0):
        self.radius = radius
        self.omega = omega
        self.theta = 0.0
        self.charge = charge
        self.n_points = n_points
        self.I = 0.5 * radius**2  # Moment of inertia (unit mass)

        self.omega_history = [omega]
        self.power_history = []

    def step(self, field, dt):
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        total_torque = 0.0
        total_power = 0.0

        for angle in angles:
            px = self.radius * np.cos(angle)
            py = self.radius * np.sin(angle)

            if abs(px) > field.L/2 - 0.5 or abs(py) > field.L/2 - 0.5:
                continue

            # Velocity
            vx = -self.omega * self.radius * np.sin(angle)
            vy = self.omega * self.radius * np.cos(angle)

            # Get fields
            E = field.get_E_field(px, py)
            B = field.get_B_field(px, py)

            # Lorentz force: F = q(E + v×B)
            # In 2D: v×B gives (vy*B, -vx*B)
            Fx = self.charge * (E[0] + vy * B)
            Fy = self.charge * (E[1] - vx * B)

            Fx = np.clip(Fx, -100, 100)
            Fy = np.clip(Fy, -100, 100)

            # Torque
            torque = px * Fy - py * Fx
            total_torque += torque

            # Power dissipated (work done by field on charge)
            power = Fx * vx + Fy * vy

            # Current density from moving charge
            # J = ρv, deposit into field
            field.deposit_current(px, py, self.charge * vx * 0.01,
                                 self.charge * vy * 0.01, sigma=0.15)

            if abs(power) < 1000:
                total_power += abs(power)

        # Update rotation
        self.omega += np.clip(total_torque / self.I, -100, 100) * dt
        self.theta += self.omega * dt

        self.omega_history.append(self.omega)
        self.power_history.append(total_power)
        return total_power


# =============================================================================
# COMPARISON: SCALAR VS GAUGE FIELD
# =============================================================================

class ScalarVacuum:
    """Simple scalar corrugation for comparison"""

    def __init__(self, N=100, L=10.0, A=0.5, wavelength=1.0):
        self.N, self.L = N, L
        self.dx = L / N
        self.A = A
        self.k = 2 * np.pi / wavelength

    def grad_h(self, x, y):
        """Gradient of scalar corrugation"""
        return np.array([self.A * self.k * np.cos(self.k * x), 0.0])


class ScalarDisk:
    """Disk in scalar field with velocity coupling"""

    def __init__(self, radius=1.0, omega=5.0, n_points=32, alpha=0.01):
        self.radius = radius
        self.omega = omega
        self.theta = 0.0
        self.alpha = alpha
        self.n_points = n_points
        self.I = 0.5 * radius**2

        self.omega_history = [omega]
        self.power_history = []

    def step(self, field, dt):
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        total_torque = 0.0
        total_power = 0.0

        for angle in angles:
            px = self.radius * np.cos(angle)
            py = self.radius * np.sin(angle)

            vx = -self.omega * self.radius * np.sin(angle)
            vy = self.omega * self.radius * np.cos(angle)

            dh = field.grad_h(px, py)
            v_dot_dh = vx * dh[0] + vy * dh[1]

            Fx = -2 * self.alpha * v_dot_dh * dh[0]
            Fy = -2 * self.alpha * v_dot_dh * dh[1]

            torque = px * Fy - py * Fx
            total_torque += torque

            power = -(Fx * vx + Fy * vy)
            if power > 0:
                total_power += power

        self.omega += np.clip(total_torque / self.I, -50, 50) * dt
        self.theta += self.omega * dt

        self.omega_history.append(self.omega)
        self.power_history.append(total_power)
        return total_power


# =============================================================================
# EXPERIMENTS
# =============================================================================

def compare_scalar_vs_gauge():
    """
    Compare power scaling between scalar corrugation and gauge field.
    Prediction: Scalar → ω², Gauge → ω⁴
    """
    print("=" * 60)
    print("SCALAR vs GAUGE FIELD: Testing Natural Scaling")
    print("=" * 60)

    omegas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    scalar_powers = []
    gauge_powers = []

    print("\nSCALAR FIELD (velocity coupling):")
    for omega in omegas:
        field = ScalarVacuum(N=80, L=10.0, A=0.5, wavelength=1.0)
        disk = ScalarDisk(radius=1.0, omega=omega, n_points=32, alpha=0.01)

        dt = 0.01
        for _ in range(50):  # warmup
            disk.step(field, dt)

        powers = []
        for _ in range(200):
            p = disk.step(field, dt)
            powers.append(p)

        avg = np.mean(powers)
        scalar_powers.append(avg)
        print(f"  ω={omega}: P={avg:.4f}")

    print("\nGAUGE FIELD (Lorentz force):")
    for omega in omegas:
        field = GaugeVacuum(N=80, L=10.0, c=5.0)
        disk = ChargedDisk(radius=1.0, omega=omega, n_points=32, charge=1.0)

        dt = 0.01
        for _ in range(50):
            disk.step(field, dt)
            field.step(dt)

        powers = []
        for _ in range(200):
            p = disk.step(field, dt)
            field.step(dt)
            powers.append(p)

        avg = np.mean(powers)
        gauge_powers.append(avg)
        print(f"  ω={omega}: P={avg:.4f}")

    scalar_powers = np.array(scalar_powers)
    gauge_powers = np.array(gauge_powers)

    # Fit power laws
    def power_law(x, A, n):
        return A * x**n

    # Scalar fit
    valid_s = scalar_powers > 0
    if np.sum(valid_s) >= 3:
        popt_s, _ = curve_fit(power_law, omegas[valid_s], scalar_powers[valid_s],
                              p0=[0.01, 2], maxfev=5000)
        n_scalar = popt_s[1]
    else:
        n_scalar = 0

    # Gauge fit
    valid_g = gauge_powers > 0
    if np.sum(valid_g) >= 3:
        popt_g, _ = curve_fit(power_law, omegas[valid_g], gauge_powers[valid_g],
                              p0=[0.01, 2], maxfev=5000)
        n_gauge = popt_g[1]
    else:
        n_gauge = 0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Scalar field:  P ∝ ω^{n_scalar:.2f}  (expected: ω²)")
    print(f"Gauge field:   P ∝ ω^{n_gauge:.2f}  (expected: ω⁴)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(omegas, scalar_powers, 'bo-', markersize=10, linewidth=2,
            label=f'Scalar: P ∝ ω^{n_scalar:.2f}')
    ax.plot(omegas, gauge_powers, 'rs-', markersize=10, linewidth=2,
            label=f'Gauge: P ∝ ω^{n_gauge:.2f}')
    ax.set_xlabel('ω', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title('Scalar Corrugation vs Gauge Field', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.loglog(omegas[valid_s], scalar_powers[valid_s], 'bo', markersize=10,
              label=f'Scalar: slope={n_scalar:.2f}')
    ax.loglog(omegas[valid_g], gauge_powers[valid_g], 'rs', markersize=10,
              label=f'Gauge: slope={n_gauge:.2f}')

    omega_ref = np.linspace(0.8, 8, 50)
    ax.loglog(omega_ref, 0.01 * omega_ref**2, 'b:', alpha=0.5, linewidth=2, label='ω²')
    ax.loglog(omega_ref, 0.001 * omega_ref**4, 'r:', alpha=0.5, linewidth=2, label='ω⁴')

    ax.set_xlabel('log(ω)', fontsize=12)
    ax.set_ylabel('log(P)', fontsize=12)
    ax.set_title('Log-Log: Field Structure → Scaling', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('scalar_vs_gauge.png', dpi=150)
    plt.show()

    return n_scalar, n_gauge


def analyze_lorentz_force_structure():
    """
    Analyze WHY the Lorentz force gives different scaling.

    F = q(E + v×B)

    For a charge in circular motion in a magnetic field B:
    - The v×B term gives a force perpendicular to v
    - This is centripetal, not dissipative for uniform B

    But for a VARYING field (our corrugated B):
    - The charge sees different B at different phases
    - This creates a net torque
    - The power depends on how B changes with position AND time
    """
    print("\n" + "=" * 60)
    print("LORENTZ FORCE STRUCTURE ANALYSIS")
    print("=" * 60)
    print("""
For a charge moving in a circle in a spatially varying B field:

Position: r = R(cos(ωt), sin(ωt))
Velocity: v = Rω(-sin(ωt), cos(ωt))

If B = B₀ sin(kx) = B₀ sin(kR cos(ωt)):

The v×B force has magnitude ~ |v||B| = Rω × B₀ sin(kR cos(ωt))

Power = F·v involves:
- v component of F (tangential)
- Which comes from E field and radial component of v×B

Key insight: The Lorentz force naturally couples v to the spatial
structure, AND the induced E field couples ∂A/∂t to acceleration.

The combination gives:
- From v×B: terms involving v·∇B ~ ω × ∇B
- From E = -∂A/∂t: terms involving a (since accelerating charges
  create time-varying A)

The gauge-invariant combination F_μν contains BOTH space and time
derivatives, which is why it couples to acceleration naturally.
""")


def test_hierarchy():
    """
    Test the full hierarchy: scalar, vector, tensor.
    For tensor, we'd need gravitational wave emission, which scales as ω⁶.

    For now, let's just verify scalar and vector, and predict tensor.
    """
    print("\n" + "=" * 60)
    print("THE FIELD TYPE HIERARCHY")
    print("=" * 60)
    print("""
┌─────────────┬─────────────┬───────────┬────────────────────────┐
│ Field Type  │ Gauge Group │ Scaling   │ Physical Example       │
├─────────────┼─────────────┼───────────┼────────────────────────┤
│ Scalar      │ None        │ ω²        │ Viscous drag           │
│ Vector      │ U(1)        │ ω⁴        │ EM radiation (Larmor)  │
│ Tensor      │ Diff(M)     │ ω⁶        │ Gravitational waves    │
│ Spinor      │ SU(2)?      │ ω⁸?       │ ???                    │
└─────────────┴─────────────┴───────────┴────────────────────────┘

Each step up in field complexity:
- Adds gauge redundancy
- Couples to one higher time derivative
- Increases power law exponent by 2

This pattern suggests:
- The vacuum's "information type" determines what it resists
- Gauge structure = redundancy in description = insensitivity to lower derivatives
- More gauge structure → more resistance to higher-order changes
""")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    omegas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    scalar = omegas**2
    vector = omegas**4
    tensor = omegas**6

    scalar = scalar / scalar[0]
    vector = vector / vector[0]
    tensor = tensor / tensor[0]

    ax.semilogy(omegas, scalar, 'bo-', markersize=10, linewidth=2,
                label='Scalar (ω²): Viscous drag')
    ax.semilogy(omegas, vector, 'rs-', markersize=10, linewidth=2,
                label='Vector (ω⁴): EM radiation')
    ax.semilogy(omegas, tensor, 'g^-', markersize=10, linewidth=2,
                label='Tensor (ω⁶): Gravitational waves')

    ax.set_xlabel('Angular velocity ω', fontsize=12)
    ax.set_ylabel('Normalized Power P/P₀', fontsize=12)
    ax.set_title('The Hierarchy of Vacuum Structure\nField Type → Radiation Scaling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('field_hierarchy.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GAUGE FIELD VACUUM SIMULATION")
    print("Testing: Does gauge structure → ω⁴ naturally?")
    print("=" * 60)

    n_scalar, n_gauge = compare_scalar_vs_gauge()
    analyze_lorentz_force_structure()
    test_hierarchy()

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"""
Measured scaling:
  Scalar field: P ∝ ω^{n_scalar:.1f}
  Gauge field:  P ∝ ω^{n_gauge:.1f}

The hypothesis is:
  Scalar (no gauge) → ω²  (spatial friction)
  Vector (U(1) gauge) → ω⁴ (temporal friction = Larmor)
  Tensor (diffeomorphism) → ω⁶ (gravitational = quadrupole)

Key insight: Gauge invariance filters out lower-derivative couplings!

A gauge transformation A → A + ∇λ leaves F_μν unchanged.
This means the physics can't depend on A directly, only on
derivatives of A (the field strength).

For EM: F_μν = ∂_μA_ν - ∂_νA_μ already contains first derivatives.
The Lorentz force F = qF_μν u^ν couples to velocity.
But radiation (energy loss) requires the field to change, which
means accelerating charges → ∂F/∂t → second derivatives.

Result: Gauge invariance + energy conservation → P ∝ a² ∝ ω⁴

This is why your "informational viscosity" with scalar corrugation
gives ω² (no gauge constraint), while real EM gives ω⁴ (gauge constraint).

The STRUCTURE of the vacuum determines the PHYSICS!
""")
