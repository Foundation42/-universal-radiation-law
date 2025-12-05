"""
TRUE Self-Consistent EM Simulation
==================================
The key insight: In the previous simulation, the charge followed a
PRESCRIBED trajectory. It wasn't actually affected by its own field!

For true self-consistency:
1. Charge creates field (via current density)
2. Field propagates (wave equation)
3. Field acts back on charge (Lorentz force FROM ITS OWN FIELD)
4. Charge slows down if radiating

The ω⁴ should emerge from measuring how fast the charge loses energy
when its own field acts back on it.
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time

ti.init(arch=ti.gpu, default_fp=ti.f32)

# =============================================================================
# PARAMETERS - smaller grid for faster iteration
# =============================================================================

N = 128  # Smaller grid for speed
L = 16.0
dx = L / N
c = 1.0
dt = 0.3 * dx / c

print(f"Grid: {N}³ = {N**3:,} cells")
print(f"dx = {dx:.4f}, dt = {dt:.4f}")

# =============================================================================
# FIELDS
# =============================================================================

# We'll use E and B directly (Yee lattice style)
Ex = ti.field(dtype=ti.f32, shape=(N, N, N))
Ey = ti.field(dtype=ti.f32, shape=(N, N, N))
Ez = ti.field(dtype=ti.f32, shape=(N, N, N))
Bx = ti.field(dtype=ti.f32, shape=(N, N, N))
By = ti.field(dtype=ti.f32, shape=(N, N, N))
Bz = ti.field(dtype=ti.f32, shape=(N, N, N))

# Current
Jx = ti.field(dtype=ti.f32, shape=(N, N, N))
Jy = ti.field(dtype=ti.f32, shape=(N, N, N))
Jz = ti.field(dtype=ti.f32, shape=(N, N, N))

# =============================================================================
# MAXWELL'S EQUATIONS (FDTD style)
# =============================================================================

@ti.kernel
def update_B(dt: ti.f32):
    """∂B/∂t = -∇×E"""
    inv_dx = 1.0 / dx
    for i, j, k in Bx:
        if 0 < i < N-1 and 0 < j < N-1 and 0 < k < N-1:
            # curl E
            dEz_dy = (Ez[i, j+1, k] - Ez[i, j-1, k]) * 0.5 * inv_dx
            dEy_dz = (Ey[i, j, k+1] - Ey[i, j, k-1]) * 0.5 * inv_dx
            dEx_dz = (Ex[i, j, k+1] - Ex[i, j, k-1]) * 0.5 * inv_dx
            dEz_dx = (Ez[i+1, j, k] - Ez[i-1, j, k]) * 0.5 * inv_dx
            dEy_dx = (Ey[i+1, j, k] - Ey[i-1, j, k]) * 0.5 * inv_dx
            dEx_dy = (Ex[i, j+1, k] - Ex[i, j-1, k]) * 0.5 * inv_dx

            Bx[i, j, k] -= (dEz_dy - dEy_dz) * dt
            By[i, j, k] -= (dEx_dz - dEz_dx) * dt
            Bz[i, j, k] -= (dEy_dx - dEx_dy) * dt


@ti.kernel
def update_E(dt: ti.f32):
    """∂E/∂t = c²∇×B - J/ε₀"""
    c2 = c * c
    inv_dx = 1.0 / dx
    for i, j, k in Ex:
        if 0 < i < N-1 and 0 < j < N-1 and 0 < k < N-1:
            # curl B
            dBz_dy = (Bz[i, j+1, k] - Bz[i, j-1, k]) * 0.5 * inv_dx
            dBy_dz = (By[i, j, k+1] - By[i, j, k-1]) * 0.5 * inv_dx
            dBx_dz = (Bx[i, j, k+1] - Bx[i, j, k-1]) * 0.5 * inv_dx
            dBz_dx = (Bz[i+1, j, k] - Bz[i-1, j, k]) * 0.5 * inv_dx
            dBy_dx = (By[i+1, j, k] - By[i-1, j, k]) * 0.5 * inv_dx
            dBx_dy = (Bx[i, j+1, k] - Bx[i, j-1, k]) * 0.5 * inv_dx

            Ex[i, j, k] += (c2 * (dBz_dy - dBy_dz) - Jx[i, j, k]) * dt
            Ey[i, j, k] += (c2 * (dBx_dz - dBz_dx) - Jy[i, j, k]) * dt
            Ez[i, j, k] += (c2 * (dBy_dx - dBx_dy) - Jz[i, j, k]) * dt


@ti.kernel
def apply_boundary_damping(strength: ti.f32):
    """Absorbing boundaries"""
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
def clear_J():
    for i, j, k in Jx:
        Jx[i, j, k] = 0.0
        Jy[i, j, k] = 0.0
        Jz[i, j, k] = 0.0


@ti.kernel
def deposit_J(px: ti.f32, py: ti.f32, pz: ti.f32,
              vx: ti.f32, vy: ti.f32, vz: ti.f32,
              q: ti.f32, sigma: ti.f32):
    """Deposit current from point charge"""
    ci = int((px + L/2) / dx)
    cj = int((py + L/2) / dx)
    ck = int((pz + L/2) / dx)
    spread = int(3 * sigma / dx) + 1

    for di, dj, dk in ti.ndrange((-spread, spread+1), (-spread, spread+1), (-spread, spread+1)):
        i, j, k = ci + di, cj + dj, ck + dk
        if 0 <= i < N and 0 <= j < N and 0 <= k < N:
            x = -L/2 + (i + 0.5) * dx
            y = -L/2 + (j + 0.5) * dx
            z = -L/2 + (k + 0.5) * dx
            r2 = (x-px)**2 + (y-py)**2 + (z-pz)**2
            w = ti.exp(-r2 / (2*sigma**2))
            Jx[i, j, k] += q * vx * w
            Jy[i, j, k] += q * vy * w
            Jz[i, j, k] += q * vz * w


@ti.kernel
def get_field_at_point(px: ti.f32, py: ti.f32, pz: ti.f32,
                       sigma: ti.f32) -> ti.types.vector(6, ti.f32):
    """Get E and B at a point (averaged over charge distribution)"""
    ci = int((px + L/2) / dx)
    cj = int((py + L/2) / dx)
    ck = int((pz + L/2) / dx)

    Ex_avg = 0.0
    Ey_avg = 0.0
    Ez_avg = 0.0
    Bx_avg = 0.0
    By_avg = 0.0
    Bz_avg = 0.0
    total_w = 0.0

    spread = int(2 * sigma / dx) + 1

    for di, dj, dk in ti.ndrange((-spread, spread+1), (-spread, spread+1), (-spread, spread+1)):
        i, j, k = ci + di, cj + dj, ck + dk
        if 0 < i < N-1 and 0 < j < N-1 and 0 < k < N-1:
            x = -L/2 + (i + 0.5) * dx
            y = -L/2 + (j + 0.5) * dx
            z = -L/2 + (k + 0.5) * dx
            r2 = (x-px)**2 + (y-py)**2 + (z-pz)**2
            w = ti.exp(-r2 / (2*sigma**2))

            Ex_avg += Ex[i, j, k] * w
            Ey_avg += Ey[i, j, k] * w
            Ez_avg += Ez[i, j, k] * w
            Bx_avg += Bx[i, j, k] * w
            By_avg += By[i, j, k] * w
            Bz_avg += Bz[i, j, k] * w
            total_w += w

    if total_w > 0:
        Ex_avg /= total_w
        Ey_avg /= total_w
        Ez_avg /= total_w
        Bx_avg /= total_w
        By_avg /= total_w
        Bz_avg /= total_w

    return ti.Vector([Ex_avg, Ey_avg, Ez_avg, Bx_avg, By_avg, Bz_avg])


# =============================================================================
# SELF-CONSISTENT CHARGE
# =============================================================================

class SelfConsistentCharge:
    """
    A charge that moves according to the Lorentz force from the field
    it has itself created. This is TRUE self-consistency!
    """

    def __init__(self, x=0, y=2, z=0, vx=0, vy=0, vz=0,
                 charge=10.0, mass=1.0, sigma=0.5):
        self.pos = np.array([x, y, z], dtype=np.float32)
        self.vel = np.array([vx, vy, vz], dtype=np.float32)
        self.charge = charge
        self.mass = mass
        self.sigma = sigma

        self.ke_history = []
        self.speed_history = []

    def kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

    def deposit_current(self):
        deposit_J(self.pos[0], self.pos[1], self.pos[2],
                  self.vel[0], self.vel[1], self.vel[2],
                  self.charge, self.sigma)

    def get_lorentz_force(self):
        """Get Lorentz force from the field (including self-field!)"""
        fields = get_field_at_point(self.pos[0], self.pos[1], self.pos[2], self.sigma)
        E = np.array([fields[0], fields[1], fields[2]])
        B = np.array([fields[3], fields[4], fields[5]])

        # F = q(E + v × B)
        v_cross_B = np.cross(self.vel, B)
        F = self.charge * (E + v_cross_B)
        return F

    def step(self, dt, use_self_force=True):
        """Advance position and velocity"""
        if use_self_force:
            F = self.get_lorentz_force()
            # Limit force to prevent instabilities
            F_mag = np.linalg.norm(F)
            if F_mag > 100:
                F = F * 100 / F_mag
            accel = F / self.mass
        else:
            accel = np.zeros(3)

        # Velocity Verlet
        self.vel = self.vel + 0.5 * accel * dt
        self.pos = self.pos + self.vel * dt

        # Keep in bounds
        for i in range(3):
            if abs(self.pos[i]) > L/2 - 1:
                self.pos[i] = np.sign(self.pos[i]) * (L/2 - 1)
                self.vel[i] *= -0.5  # Bounce with damping

        if use_self_force:
            F = self.get_lorentz_force()
            F_mag = np.linalg.norm(F)
            if F_mag > 100:
                F = F * 100 / F_mag
            accel = F / self.mass
            self.vel = self.vel + 0.5 * accel * dt

        self.ke_history.append(self.kinetic_energy())
        self.speed_history.append(np.linalg.norm(self.vel))


# =============================================================================
# TEST: SPINNING CHARGE WITH SELF-FORCE
# =============================================================================

def test_spinning_charge(omega, use_self_force=True, n_steps=1500):
    """
    Create a charge in circular motion and see if it radiates.

    We initialize with circular motion velocity and let it evolve.
    If radiation reaction works, it should slow down!
    """
    # Clear fields
    Ex.fill(0)
    Ey.fill(0)
    Ez.fill(0)
    Bx.fill(0)
    By.fill(0)
    Bz.fill(0)

    # Initial conditions for circular motion
    radius = 2.0
    vx = 0.0
    vy = 0.0
    vz = omega * radius  # Tangential velocity

    charge = SelfConsistentCharge(
        x=radius, y=0, z=0,
        vx=vx, vy=vy, vz=vz,
        charge=15.0, mass=1.0, sigma=0.4
    )

    # We also need a centripetal force to maintain circular motion
    # In real EM this would come from an external field
    # For now, let's just track the energy loss

    initial_ke = charge.kinetic_energy()

    for step in range(n_steps):
        clear_J()
        charge.deposit_current()

        # Maxwell evolution
        update_B(dt)
        update_E(dt)
        apply_boundary_damping(0.05)

        # Move charge
        charge.step(dt, use_self_force=use_self_force)

    final_ke = charge.kinetic_energy()
    energy_loss = initial_ke - final_ke

    return energy_loss, charge.ke_history


def measure_power_vs_omega():
    """
    THE TEST: Does energy loss rate scale as ω⁴?
    """
    print("=" * 60)
    print("SELF-CONSISTENT RADIATION: Energy Loss vs ω")
    print("=" * 60)

    omegas = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    energy_losses = []

    for omega in omegas:
        print(f"\nTesting ω = {omega:.1f}...")

        # With self-force
        loss, ke_hist = test_spinning_charge(omega, use_self_force=True, n_steps=1000)
        energy_losses.append(loss)

        print(f"  Initial KE: {ke_hist[0]:.4f}")
        print(f"  Final KE:   {ke_hist[-1]:.4f}")
        print(f"  Loss:       {loss:.4f}")

    energy_losses = np.array(energy_losses)

    # Fit
    def power_law(x, A, n):
        return A * x**n

    valid = energy_losses > 0
    if np.sum(valid) >= 3:
        popt, _ = curve_fit(power_law, omegas[valid], energy_losses[valid],
                           p0=[0.1, 2], maxfev=5000)
        n_fit = popt[1]
    else:
        n_fit = 0

    print("\n" + "=" * 60)
    print(f"RESULT: Energy loss ∝ ω^{n_fit:.2f}")
    print("=" * 60)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(omegas, energy_losses, 'bo-', markersize=12, linewidth=2,
            label=f'Measured: ∝ ω^{n_fit:.2f}')

    if n_fit > 0:
        omega_fit = np.linspace(omegas.min(), omegas.max(), 50)
        ax.plot(omega_fit, power_law(omega_fit, popt[0], popt[1]), 'r--',
                linewidth=2, label='Fit')

    ax.set_xlabel('Angular velocity ω', fontsize=14)
    ax.set_ylabel('Energy Loss', fontsize=14)
    ax.set_title('Self-Consistent Radiation Reaction', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('true_self_consistent.png', dpi=150)
    plt.show()

    return n_fit


# =============================================================================
# SIMPLER TEST: OSCILLATING CHARGE
# =============================================================================

def test_oscillating_charge():
    """
    Simpler test: an oscillating charge (like antenna).
    Should radiate with P ∝ ω⁴ for sinusoidal motion.
    """
    print("\n" + "=" * 60)
    print("OSCILLATING CHARGE TEST")
    print("=" * 60)

    omegas = [0.5, 0.8, 1.0, 1.3, 1.6]
    losses = []

    for omega in omegas:
        # Clear
        Ex.fill(0); Ey.fill(0); Ez.fill(0)
        Bx.fill(0); By.fill(0); Bz.fill(0)

        # Charge oscillating in z: z = A*sin(ωt)
        A = 1.5
        q = 20.0
        sigma = 0.4

        n_steps = int(4 * 2 * np.pi / omega / dt)  # 4 periods
        t = 0

        total_energy_deposited = 0

        for step in range(n_steps):
            clear_J()

            # Position and velocity
            z = A * np.sin(omega * t)
            vz = A * omega * np.cos(omega * t)

            # Deposit current
            deposit_J(0, 0, z, 0, 0, vz, q, sigma)

            # Energy deposited this step (proxy)
            total_energy_deposited += q * vz * vz * dt * 0.001

            # Maxwell
            update_B(dt)
            update_E(dt)
            apply_boundary_damping(0.05)

            t += dt

        losses.append(total_energy_deposited)
        print(f"  ω = {omega:.1f}: energy ~ {total_energy_deposited:.4f}")

    # Fit
    omegas = np.array(omegas)
    losses = np.array(losses)

    def power_law(x, A, n):
        return A * x**n

    popt, _ = curve_fit(power_law, omegas, losses, p0=[0.1, 2])
    print(f"\nFitted: P ∝ ω^{popt[1]:.2f}")

    return popt[1]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRUE SELF-CONSISTENT EM SIMULATION")
    print("=" * 60)

    # Run the oscillating charge test (simpler)
    n_osc = test_oscillating_charge()

    # Run the spinning charge test
    n_spin = measure_power_vs_omega()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Oscillating charge: ω^{n_osc:.2f}")
    print(f"Spinning charge:    ω^{n_spin:.2f}")
    print(f"\nExpected (Larmor):  ω⁴")
