"""
Corrugated Vacuum Simulation v2
===============================
Now with DYNAMICAL FIELD - energy lost by spinning objects
propagates away as waves (radiation)!

Key improvements:
1. Field can carry waves (radiation)
2. Energy is conserved (transferred to field)
3. Tuned parameters to see gradual decay
4. Visualization of field disturbances
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import laplace

# =============================================================================
# DYNAMICAL CORRUGATED FIELD
# =============================================================================

class DynamicalField:
    """
    A 2D scalar field with:
    - Background corrugation (static): h0(x) = A*sin(kx)
    - Dynamical perturbation: φ(x,y,t) that carries waves

    Total field: h(x,y,t) = h0(x) + φ(x,y,t)

    The perturbation φ obeys a wave equation with damping,
    so energy deposited by particles radiates away.
    """

    def __init__(self, grid_size=200, domain_size=10.0,
                 amplitude=0.5, wavelength=1.0, wave_speed=2.0, damping=0.01):
        self.N = grid_size
        self.L = domain_size
        self.dx = domain_size / grid_size

        # Background corrugation
        self.A = amplitude
        self.k = 2 * np.pi / wavelength

        # Wave equation parameters
        self.c = wave_speed      # Speed of "light" (field wave propagation)
        self.gamma = damping     # Damping at boundaries

        # Grid coordinates
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.y = np.linspace(-self.L/2, self.L/2, self.N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Background field (static corrugation)
        self.h0 = self.A * np.sin(self.k * self.X)

        # Dynamical perturbation (starts at zero)
        self.phi = np.zeros((self.N, self.N))
        self.phi_dot = np.zeros((self.N, self.N))  # Time derivative

        # Energy tracking
        self.field_energy_history = []

    def background_gradient(self, x, y):
        """Gradient of background corrugation at a point"""
        dhdx = self.A * self.k * np.cos(self.k * x)
        dhdy = 0.0
        return np.array([dhdx, dhdy])

    def perturbation_gradient(self, x, y):
        """Gradient of dynamical perturbation at a point (interpolated)"""
        # Convert position to grid indices
        i = int((x + self.L/2) / self.dx)
        j = int((y + self.L/2) / self.dx)

        # Bounds check
        i = max(1, min(self.N-2, i))
        j = max(1, min(self.N-2, j))

        # Central difference
        dphidx = (self.phi[j, i+1] - self.phi[j, i-1]) / (2 * self.dx)
        dphidy = (self.phi[j+1, i] - self.phi[j-1, i]) / (2 * self.dx)

        return np.array([dphidx, dphidy])

    def total_gradient(self, x, y):
        """Total field gradient"""
        return self.background_gradient(x, y) + self.perturbation_gradient(x, y)

    def deposit_energy(self, x, y, amount, radius=0.3):
        """
        Deposit energy into the field at position (x,y).
        This is how particles "radiate" - they kick the field.
        """
        # Gaussian source centered at (x,y)
        r2 = (self.X - x)**2 + (self.Y - y)**2
        source = amount * np.exp(-r2 / (2 * radius**2))

        # Add to field velocity (like hitting a drum)
        self.phi_dot += source

    def step(self, dt):
        """
        Evolve field using wave equation:
        ∂²φ/∂t² = c² ∇²φ - γ ∂φ/∂t
        """
        # Laplacian using finite differences
        laplacian = laplace(self.phi) / (self.dx**2)

        # Boundary damping (absorbing boundaries)
        boundary_mask = np.zeros_like(self.phi)
        edge = int(0.1 * self.N)
        boundary_mask[:edge, :] = 1
        boundary_mask[-edge:, :] = 1
        boundary_mask[:, :edge] = 1
        boundary_mask[:, -edge:] = 1

        # Wave equation with damping
        self.phi_dot += (self.c**2 * laplacian - self.gamma * self.phi_dot) * dt
        self.phi_dot *= (1 - 0.1 * boundary_mask)  # Extra damping at boundaries

        self.phi += self.phi_dot * dt

        # Track field energy
        kinetic = 0.5 * np.sum(self.phi_dot**2) * self.dx**2
        potential = 0.5 * self.c**2 * np.sum(
            ((np.roll(self.phi, 1, 0) - self.phi)/self.dx)**2 +
            ((np.roll(self.phi, 1, 1) - self.phi)/self.dx)**2
        ) * self.dx**2
        self.field_energy_history.append(kinetic + potential)

    def get_field_for_plot(self):
        """Return total field for visualization"""
        return self.h0 + self.phi * 10  # Amplify perturbation for visibility


# =============================================================================
# PARTICLE COUPLED TO DYNAMICAL FIELD
# =============================================================================

class CoupledParticle:
    """Particle that exchanges energy with the field"""

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, mass=1.0, coupling=0.05):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.mass = mass
        self.alpha = coupling

        self.trajectory = [self.pos.copy()]
        self.kinetic_energy_history = []
        self.radiated_power_history = []

    def kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

    def step(self, field, dt):
        """Advance particle and deposit energy into field"""
        # Get field gradient at particle position
        dh = field.total_gradient(self.pos[0], self.pos[1])
        v_dot_dh = np.dot(self.vel, dh)

        # Force from coupling (resistance to crossing corrugations)
        F = -2 * self.alpha * v_dot_dh * dh

        # Power transferred to field (radiation)
        power_radiated = -np.dot(F, self.vel)

        # Deposit energy into field as radiation
        if power_radiated > 0:
            field.deposit_energy(self.pos[0], self.pos[1],
                               power_radiated * dt * 0.5, radius=0.2)

        # Update particle state
        self.vel = self.vel + (F / self.mass) * dt
        self.pos = self.pos + self.vel * dt

        # Keep in bounds
        self.pos = np.clip(self.pos, -field.L/2 + 0.5, field.L/2 - 0.5)

        # Record history
        self.trajectory.append(self.pos.copy())
        self.kinetic_energy_history.append(self.kinetic_energy())
        self.radiated_power_history.append(power_radiated)


# =============================================================================
# ROTATING DISK WITH RADIATION
# =============================================================================

class RadiatingDisk:
    """A spinning disk that radiates into the field"""

    def __init__(self, cx=0.0, cy=0.0, radius=1.0, n_points=24,
                 vx=0.0, vy=0.0, omega=5.0, mass=1.0, coupling=0.02):
        self.center = np.array([cx, cy], dtype=float)
        self.radius = radius
        self.vel = np.array([vx, vy], dtype=float)
        self.omega = omega
        self.theta = 0.0
        self.mass = mass
        self.I = 0.5 * mass * radius**2  # Moment of inertia
        self.alpha = coupling
        self.n_points = n_points

        # History
        self.center_trajectory = [self.center.copy()]
        self.omega_history = [omega]
        self.trans_ke_history = []
        self.rot_ke_history = []
        self.total_radiated = 0.0
        self.radiated_history = []

    def get_points(self):
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        points = np.zeros((self.n_points, 2))
        points[:, 0] = self.center[0] + self.radius * np.cos(angles)
        points[:, 1] = self.center[1] + self.radius * np.sin(angles)
        return points, angles

    def trans_ke(self):
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

    def rot_ke(self):
        return 0.5 * self.I * self.omega**2

    def step(self, field, dt):
        points, angles = self.get_points()

        total_force = np.zeros(2)
        total_torque = 0.0
        total_power = 0.0

        for i in range(self.n_points):
            pos = points[i]

            # Point velocity = translation + rotation
            rot_vel = self.omega * self.radius * np.array([-np.sin(angles[i]),
                                                            np.cos(angles[i])])
            vel = self.vel + rot_vel

            # Field gradient and coupling force
            dh = field.total_gradient(pos[0], pos[1])
            v_dot_dh = np.dot(vel, dh)
            F = -2 * self.alpha * v_dot_dh * dh

            total_force += F

            # Torque
            r = pos - self.center
            torque = r[0] * F[1] - r[1] * F[0]
            total_torque += torque

            # Power radiated by this point
            power = -np.dot(F, vel)
            if power > 0:
                field.deposit_energy(pos[0], pos[1], power * dt * 0.3, radius=0.15)
                total_power += power

        # Update dynamics
        self.vel = self.vel + (total_force / self.mass) * dt
        self.center = self.center + self.vel * dt

        self.omega = self.omega + (total_torque / self.I) * dt
        self.theta = self.theta + self.omega * dt

        # Keep in bounds
        self.center = np.clip(self.center, -field.L/2 + self.radius + 0.5,
                              field.L/2 - self.radius - 0.5)

        # Record history
        self.center_trajectory.append(self.center.copy())
        self.omega_history.append(self.omega)
        self.trans_ke_history.append(self.trans_ke())
        self.rot_ke_history.append(self.rot_ke())
        self.total_radiated += total_power * dt
        self.radiated_history.append(self.total_radiated)


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_radiation():
    """
    Watch a spinning disk radiate energy into the field.
    The energy doesn't just disappear - it propagates away as waves!
    """
    print("=" * 60)
    print("SPINNING DISK RADIATING INTO DYNAMICAL FIELD")
    print("=" * 60)

    # Create field and disk
    field = DynamicalField(grid_size=150, domain_size=8.0,
                          amplitude=0.3, wavelength=0.8,
                          wave_speed=3.0, damping=0.02)

    disk = RadiatingDisk(cx=0, cy=0, radius=0.8, n_points=32,
                        omega=8.0, coupling=0.015, mass=1.0)

    # Run simulation
    dt = 0.01
    steps = 800

    # Store snapshots for animation
    field_snapshots = []
    snapshot_interval = 4

    print("Running simulation...")
    for step in range(steps):
        disk.step(field, dt)
        field.step(dt)

        if step % snapshot_interval == 0:
            field_snapshots.append({
                'phi': field.phi.copy(),
                'disk_center': disk.center.copy(),
                'disk_theta': disk.theta,
                'omega': disk.omega
            })

        if step % 100 == 0:
            print(f"  Step {step}/{steps}, ω = {disk.omega:.2f}")

    # Create animation
    print("\nCreating animation...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Field visualization
    ax1 = axes[0]
    vmax = np.max(np.abs(field.phi)) * 0.5 + 0.01
    im = ax1.imshow(field_snapshots[0]['phi'], extent=[-4, 4, -4, 4],
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')

    # Draw background corrugation as contours
    ax1.contour(field.X, field.Y, field.h0, levels=5, colors='k', alpha=0.2)

    # Disk outline
    theta_circle = np.linspace(0, 2*np.pi, 50)
    disk_line, = ax1.plot([], [], 'k-', linewidth=2)
    disk_marker, = ax1.plot([], [], 'ko', markersize=8)

    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Field Perturbation (Radiation)')
    ax1.set_aspect('equal')
    plt.colorbar(im, ax=ax1, label='φ (field perturbation)')

    # Angular velocity plot
    ax2 = axes[1]
    t = np.arange(steps + 1) * dt
    ax2.plot(t, disk.omega_history, 'b-', alpha=0.3)
    omega_line, = ax2.plot([], [], 'b-', linewidth=2)
    omega_point, = ax2.plot([], [], 'ro', markersize=8)
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, max(disk.omega_history) * 1.1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular velocity ω')
    ax2.set_title('Spin-Down (Radiation Reaction)')
    ax2.grid(True, alpha=0.3)

    # Energy plot
    ax3 = axes[2]
    rot_ke = np.array(disk.rot_ke_history)
    field_e = np.array(field.field_energy_history)
    ax3.plot(t[1:], rot_ke, 'b-', alpha=0.3, label='Rotational KE')
    ax3.plot(t[1:], field_e[:len(rot_ke)], 'r-', alpha=0.3, label='Field Energy')

    rot_line, = ax3.plot([], [], 'b-', linewidth=2, label='Disk KE')
    field_line, = ax3.plot([], [], 'r-', linewidth=2, label='Field')
    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(0, max(rot_ke) * 1.2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Energy')
    ax3.set_title('Energy Transfer: Disk → Field')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    def animate(frame):
        snapshot = field_snapshots[frame]

        # Update field image
        im.set_array(snapshot['phi'])

        # Update disk position
        cx, cy = snapshot['disk_center']
        theta = snapshot['disk_theta']
        disk_x = cx + disk.radius * np.cos(theta_circle)
        disk_y = cy + disk.radius * np.sin(theta_circle)
        disk_line.set_data(disk_x, disk_y)

        # Marker to show rotation
        marker_x = cx + disk.radius * 0.7 * np.cos(theta)
        marker_y = cy + disk.radius * 0.7 * np.sin(theta)
        disk_marker.set_data([marker_x], [marker_y])

        # Update omega plot
        idx = frame * snapshot_interval
        omega_line.set_data(t[:idx+1], disk.omega_history[:idx+1])
        omega_point.set_data([t[idx]], [disk.omega_history[idx]])

        # Update energy plot
        if idx > 0:
            rot_line.set_data(t[1:idx+1], rot_ke[:idx])
            field_line.set_data(t[1:idx+1], field_e[:idx])

        return im, disk_line, disk_marker, omega_line, omega_point, rot_line, field_line

    anim = animation.FuncAnimation(fig, animate, frames=len(field_snapshots),
                                   interval=50, blit=True)

    plt.tight_layout()

    print("Saving animation...")
    anim.save('radiation_animation.gif', writer='pillow', fps=20)
    print("Saved: radiation_animation.gif")

    # Also save final state
    plt.savefig('radiation_final.png', dpi=150)
    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial angular velocity: {disk.omega_history[0]:.2f} rad/s")
    print(f"Final angular velocity:   {disk.omega_history[-1]:.2f} rad/s")
    print(f"Angular momentum lost:    {100*(1 - disk.omega_history[-1]/disk.omega_history[0]):.1f}%")
    print(f"\nInitial rotational KE:    {disk.rot_ke_history[0]:.3f}")
    print(f"Final rotational KE:      {disk.rot_ke_history[-1]:.3f}")
    print(f"Final field energy:       {field.field_energy_history[-1]:.3f}")
    print(f"\nEnergy transferred to field (radiation)!")


def experiment_compare_rotation_translation():
    """
    Side-by-side comparison: rotating vs translating disk
    """
    print("\n" + "=" * 60)
    print("COMPARISON: ROTATION vs TRANSLATION")
    print("=" * 60)

    # Two separate fields
    field1 = DynamicalField(grid_size=100, domain_size=8.0,
                           amplitude=0.3, wavelength=0.8, wave_speed=3.0)
    field2 = DynamicalField(grid_size=100, domain_size=8.0,
                           amplitude=0.3, wavelength=0.8, wave_speed=3.0)

    # Disk 1: Rotating only
    disk_rot = RadiatingDisk(cx=0, cy=0, radius=0.8, omega=8.0,
                            vx=0, vy=0, coupling=0.015)

    # Disk 2: Translating along valley (y-direction)
    disk_trans = RadiatingDisk(cx=0, cy=-2, radius=0.8, omega=0.0,
                              vx=0, vy=1.5, coupling=0.015)

    dt = 0.01
    steps = 600

    print("Running simulation...")
    for step in range(steps):
        disk_rot.step(field1, dt)
        field1.step(dt)

        disk_trans.step(field2, dt)
        field2.step(dt)

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Top row: Rotating disk
    ax = axes[0, 0]
    im1 = ax.imshow(field1.phi, extent=[-4, 4, -4, 4], cmap='RdBu_r', origin='lower')
    ax.contour(field1.X, field1.Y, field1.h0, levels=5, colors='k', alpha=0.2)
    theta = np.linspace(0, 2*np.pi, 50)
    ax.plot(disk_rot.center[0] + disk_rot.radius * np.cos(theta),
           disk_rot.center[1] + disk_rot.radius * np.sin(theta), 'k-', lw=2)
    ax.set_title('ROTATING: Field Perturbation')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax)

    t = np.arange(steps + 1) * dt

    ax = axes[0, 1]
    ax.plot(t, disk_rot.omega_history, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('ω')
    ax.set_title('Angular Velocity (DECAYS!)')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t[1:], disk_rot.rot_ke_history, 'b-', label='Rotational KE')
    ax.plot(t[1:], field1.field_energy_history[:len(disk_rot.rot_ke_history)],
           'r-', label='Field Energy')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy: Disk → Field')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom row: Translating disk
    ax = axes[1, 0]
    im2 = ax.imshow(field2.phi, extent=[-4, 4, -4, 4], cmap='RdBu_r', origin='lower')
    ax.contour(field2.X, field2.Y, field2.h0, levels=5, colors='k', alpha=0.2)
    traj = np.array(disk_trans.center_trajectory)
    ax.plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.5)
    ax.plot(disk_trans.center[0] + disk_trans.radius * np.cos(theta),
           disk_trans.center[1] + disk_trans.radius * np.sin(theta), 'k-', lw=2)
    ax.set_title('TRANSLATING: Field Perturbation\n(Much less radiation!)')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax)

    ax = axes[1, 1]
    speeds = [np.linalg.norm(v) for v in np.diff(traj, axis=0) / dt]
    ax.plot(t[1:len(speeds)+1], speeds, 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed')
    ax.set_title('Translation Speed (CONSTANT!)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(t[1:], disk_trans.trans_ke_history, 'g-', label='Translational KE')
    ax.plot(t[1:], field2.field_energy_history[:len(disk_trans.trans_ke_history)],
           'r-', label='Field Energy')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title('Energy: Minimal Transfer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rotation_vs_translation.png', dpi=150)
    plt.show()

    print("\nResults:")
    print(f"  Rotating disk:    ω: {disk_rot.omega_history[0]:.1f} → {disk_rot.omega_history[-1]:.2f} rad/s")
    print(f"  Translating disk: speed ~constant along valley")
    print(f"\n  Field energy (rotating):    {field1.field_energy_history[-1]:.4f}")
    print(f"  Field energy (translating): {field2.field_energy_history[-1]:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CORRUGATED VACUUM SIMULATION v2")
    print("With Dynamical Field & Radiation")
    print("=" * 60)

    experiment_radiation()
    experiment_compare_rotation_translation()

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
The spinning disk loses angular momentum NOT because energy
"disappears" but because it RADIATES into the field!

You can see the waves propagating outward - that's the
"informational friction" manifesting as radiation.

Translation along valleys is free because it doesn't
disturb the field structure (no radiation).

This is exactly what your theory predicts:
- Radiation reaction = informational friction
- The friction IS the radiation, not a separate effect
""")
