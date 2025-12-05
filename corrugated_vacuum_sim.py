"""
Corrugated Vacuum Simulation
============================
Exploring "informational viscosity" - a medium where linear motion is free
but rotation/acceleration costs energy (manifests as radiation/drag).

The vacuum is modeled as a corrugated surface: h(x,y) = A*sin(kx)
- Motion along valleys (y-direction): frictionless
- Motion across ridges (x-direction): resisted
- Rotation: always crosses ridges -> drag
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

# =============================================================================
# CORRUGATED FIELD
# =============================================================================

class CorrugatedField:
    """The vacuum's corrugation structure"""

    def __init__(self, amplitude=1.0, wavelength=1.0):
        self.A = amplitude
        self.k = 2 * np.pi / wavelength

    def height(self, x, y):
        """Height of corrugation at position (x, y)"""
        return self.A * np.sin(self.k * x)

    def gradient(self, x, y):
        """Gradient of height field: (dh/dx, dh/dy)"""
        dhdx = self.A * self.k * np.cos(self.k * x)
        dhdy = 0.0
        return np.array([dhdx, dhdy])

    def plot(self, ax, xlim=(-3, 3), ylim=(-3, 3), resolution=100):
        """Visualize the corrugation as a contour plot"""
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = self.height(X, Y)

        # Contour plot showing valleys and ridges
        contour = ax.contourf(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.6)
        ax.contour(X, Y, Z, levels=10, colors='k', alpha=0.3, linewidths=0.5)
        return contour


# =============================================================================
# SINGLE PARTICLE
# =============================================================================

class Particle:
    """A point particle coupled to the corrugated vacuum"""

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, mass=1.0, coupling=0.1):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.mass = mass
        self.alpha = coupling  # Strength of coupling to corrugation
        self.trajectory = [self.pos.copy()]
        self.energy_history = []

    def kinetic_energy(self):
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

    def coupling_energy(self, field):
        """Energy from coupling to field gradient"""
        dh = field.gradient(self.pos[0], self.pos[1])
        v_dot_dh = np.dot(self.vel, dh)
        return self.alpha * v_dot_dh**2

    def total_energy(self, field):
        return self.kinetic_energy() + self.coupling_energy(field)

    def step(self, field, dt):
        """Advance particle by one timestep"""
        dh = field.gradient(self.pos[0], self.pos[1])
        v_dot_dh = np.dot(self.vel, dh)

        # Force from coupling term: F = -∇[α(v·∇h)²]
        # This creates resistance to motion across corrugations
        F = -2 * self.alpha * v_dot_dh * dh

        # Update velocity and position (simple Euler for now)
        self.vel = self.vel + (F / self.mass) * dt
        self.pos = self.pos + self.vel * dt

        self.trajectory.append(self.pos.copy())
        self.energy_history.append(self.total_energy(field))


# =============================================================================
# ROTATING DISK (collection of particles)
# =============================================================================

class RotatingDisk:
    """A rigid disk that can translate and rotate"""

    def __init__(self, cx=0.0, cy=0.0, radius=0.5, n_points=12,
                 vx=0.0, vy=0.0, omega=0.0, mass=1.0, coupling=0.1):
        self.center = np.array([cx, cy], dtype=float)
        self.radius = radius
        self.vel = np.array([vx, vy], dtype=float)  # Center of mass velocity
        self.omega = omega  # Angular velocity (rad/s)
        self.theta = 0.0    # Current rotation angle
        self.mass = mass
        self.moment_of_inertia = 0.5 * mass * radius**2
        self.alpha = coupling
        self.n_points = n_points

        # History tracking
        self.center_trajectory = [self.center.copy()]
        self.omega_history = [omega]
        self.kinetic_energy_history = []
        self.rotational_energy_history = []

    def get_point_positions(self):
        """Get positions of all points on the disk rim"""
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        points = np.zeros((self.n_points, 2))
        points[:, 0] = self.center[0] + self.radius * np.cos(angles)
        points[:, 1] = self.center[1] + self.radius * np.sin(angles)
        return points

    def get_point_velocities(self):
        """Get velocities of all points (translation + rotation)"""
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        velocities = np.zeros((self.n_points, 2))
        for i, angle in enumerate(angles):
            # Rotational velocity component (perpendicular to radius)
            rot_vel = self.omega * self.radius * np.array([-np.sin(angle), np.cos(angle)])
            velocities[i] = self.vel + rot_vel
        return velocities

    def translational_ke(self):
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

    def rotational_ke(self):
        return 0.5 * self.moment_of_inertia * self.omega**2

    def step(self, field, dt):
        """Advance disk by one timestep"""
        points = self.get_point_positions()
        velocities = self.get_point_velocities()

        # Calculate total force and torque from field coupling
        total_force = np.zeros(2)
        total_torque = 0.0

        for i in range(self.n_points):
            pos = points[i]
            vel = velocities[i]
            dh = field.gradient(pos[0], pos[1])
            v_dot_dh = np.dot(vel, dh)

            # Force on this point
            F = -2 * self.alpha * v_dot_dh * dh
            total_force += F

            # Torque from this force about center
            r = pos - self.center
            torque = r[0] * F[1] - r[1] * F[0]  # Cross product in 2D
            total_torque += torque

        # Update center of mass
        accel = total_force / self.mass
        self.vel = self.vel + accel * dt
        self.center = self.center + self.vel * dt

        # Update rotation
        angular_accel = total_torque / self.moment_of_inertia
        self.omega = self.omega + angular_accel * dt
        self.theta = self.theta + self.omega * dt

        # Record history
        self.center_trajectory.append(self.center.copy())
        self.omega_history.append(self.omega)
        self.kinetic_energy_history.append(self.translational_ke())
        self.rotational_energy_history.append(self.rotational_ke())


# =============================================================================
# SIMULATION EXPERIMENTS
# =============================================================================

def experiment_1_single_particles():
    """
    Compare particles moving along vs across corrugations.
    Key prediction: motion along valleys (y) is free, across (x) is resisted.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Single Particles - Along vs Across Corrugations")
    print("=" * 60)

    field = CorrugatedField(amplitude=1.0, wavelength=1.0)

    # Particle A: moving along valleys (y-direction)
    particle_along = Particle(x=0.25, y=-2, vx=0, vy=2.0, coupling=0.5)

    # Particle B: moving across valleys (x-direction)
    particle_across = Particle(x=-2, y=0, vx=2.0, vy=0, coupling=0.5)

    # Particle C: diagonal motion
    particle_diag = Particle(x=-2, y=-2, vx=1.4, vy=1.4, coupling=0.5)

    particles = [particle_along, particle_across, particle_diag]
    labels = ['Along valleys (vy only)', 'Across ridges (vx only)', 'Diagonal']
    colors = ['green', 'red', 'blue']

    # Run simulation
    dt = 0.005
    steps = 1000

    for _ in range(steps):
        for p in particles:
            p.step(field, dt)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Trajectories on field
    ax1 = axes[0]
    field.plot(ax1, xlim=(-3, 3), ylim=(-3, 3))
    for p, label, color in zip(particles, labels, colors):
        traj = np.array(p.trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, label=label)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)
        ax1.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Trajectories on Corrugated Vacuum')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')

    # Velocity magnitude over time
    ax2 = axes[1]
    t = np.arange(steps + 1) * dt
    for p, label, color in zip(particles, labels, colors):
        traj = np.array(p.trajectory)
        # Calculate velocities from trajectory
        vel = np.diff(traj, axis=0) / dt
        speed = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        ax2.plot(t[:-1], speed, color=color, label=label)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Speed')
    ax2.set_title('Speed vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Energy over time
    ax3 = axes[2]
    for p, label, color in zip(particles, labels, colors):
        ax3.plot(t[1:], p.energy_history, color=color, label=label)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Total Energy')
    ax3.set_title('Energy Conservation Check')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment1_particles.png', dpi=150)
    plt.show()

    print("\nResults:")
    for p, label in zip(particles, labels):
        traj = np.array(p.trajectory)
        final_speed = np.linalg.norm(p.vel)
        print(f"  {label}: final speed = {final_speed:.3f}")


def experiment_2_rotation_vs_translation():
    """
    Compare a translating disk vs a rotating disk.
    Key prediction: rotation should experience drag, translation should not.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Disk - Rotation vs Translation")
    print("=" * 60)

    field = CorrugatedField(amplitude=1.0, wavelength=1.0)

    # Disk A: pure translation along valley
    disk_translate = RotatingDisk(cx=0, cy=-3, radius=0.5, n_points=24,
                                   vx=0, vy=2.0, omega=0.0, coupling=0.3)

    # Disk B: pure rotation (stationary center)
    disk_rotate = RotatingDisk(cx=0, cy=0, radius=0.5, n_points=24,
                                vx=0, vy=0, omega=10.0, coupling=0.3)

    # Disk C: translation + rotation
    disk_both = RotatingDisk(cx=0, cy=3, radius=0.5, n_points=24,
                              vx=0, vy=-2.0, omega=10.0, coupling=0.3)

    disks = [disk_translate, disk_rotate, disk_both]
    labels = ['Translation only', 'Rotation only', 'Both']
    colors = ['green', 'red', 'blue']

    # Run simulation
    dt = 0.002
    steps = 2000

    for _ in range(steps):
        for d in disks:
            d.step(field, dt)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectories
    ax1 = axes[0, 0]
    field.plot(ax1, xlim=(-2, 2), ylim=(-5, 5))
    for d, label, color in zip(disks, labels, colors):
        traj = np.array(d.center_trajectory)
        ax1.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, label=label)
        # Draw initial disk
        circle = Circle(traj[0], d.radius, fill=False, color=color, linestyle='--')
        ax1.add_patch(circle)
        # Draw final disk
        circle = Circle(traj[-1], d.radius, fill=False, color=color, linewidth=2)
        ax1.add_patch(circle)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Disk Center Trajectories')
    ax1.legend()
    ax1.set_aspect('equal')

    # Angular velocity over time
    ax2 = axes[0, 1]
    t = np.arange(steps + 1) * dt
    for d, label, color in zip(disks, labels, colors):
        ax2.plot(t, d.omega_history, color=color, label=label)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular velocity ω (rad/s)')
    ax2.set_title('Angular Velocity vs Time\n(Rotation should decay!)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Translational KE
    ax3 = axes[1, 0]
    for d, label, color in zip(disks, labels, colors):
        ax3.plot(t[1:], d.kinetic_energy_history, color=color, label=label)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Translational KE')
    ax3.set_title('Translational Kinetic Energy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Rotational KE
    ax4 = axes[1, 1]
    for d, label, color in zip(disks, labels, colors):
        ax4.plot(t[1:], d.rotational_energy_history, color=color, label=label)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Rotational KE')
    ax4.set_title('Rotational Kinetic Energy\n(Should decay for rotating disks!)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment2_rotation.png', dpi=150)
    plt.show()

    print("\nResults:")
    for d, label in zip(disks, labels):
        print(f"  {label}:")
        print(f"    Initial ω = {d.omega_history[0]:.2f}, Final ω = {d.omega_history[-1]:.2f}")
        print(f"    ω decay = {100*(1 - d.omega_history[-1]/max(0.001, d.omega_history[0])):.1f}%")


def experiment_3_coupling_strength():
    """
    Vary the coupling strength (informational viscosity).
    Shows how stronger coupling = more friction = faster spin-down.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Effect of Coupling Strength (Viscosity)")
    print("=" * 60)

    field = CorrugatedField(amplitude=1.0, wavelength=1.0)

    coupling_values = [0.1, 0.3, 0.5, 1.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(coupling_values)))

    disks = []
    for alpha in coupling_values:
        disk = RotatingDisk(cx=0, cy=0, radius=0.5, n_points=24,
                            vx=0, vy=0, omega=10.0, coupling=alpha)
        disks.append(disk)

    # Run simulation
    dt = 0.002
    steps = 2500

    for _ in range(steps):
        for d in disks:
            d.step(field, dt)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    t = np.arange(steps + 1) * dt

    ax1 = axes[0]
    for d, alpha, color in zip(disks, coupling_values, colors):
        ax1.plot(t, d.omega_history, color=color, label=f'α = {alpha}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Angular velocity ω')
    ax1.set_title('Spin-down vs Coupling Strength\nHigher α = more "informational viscosity"')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for d, alpha, color in zip(disks, coupling_values, colors):
        ax2.plot(t[1:], d.rotational_energy_history, color=color, label=f'α = {alpha}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Rotational KE')
    ax2.set_title('Rotational Energy Decay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment3_coupling.png', dpi=150)
    plt.show()

    print("\nDecay rates:")
    for d, alpha in zip(disks, coupling_values):
        decay_percent = 100 * (1 - d.omega_history[-1] / d.omega_history[0])
        print(f"  α = {alpha}: {decay_percent:.1f}% decay in ω")


def experiment_4_animation():
    """
    Create an animation showing a spinning disk losing angular momentum.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Animation of Spinning Disk")
    print("=" * 60)

    field = CorrugatedField(amplitude=1.0, wavelength=0.8)
    disk = RotatingDisk(cx=0, cy=0, radius=0.8, n_points=36,
                        vx=0, vy=0, omega=15.0, coupling=0.4)

    # Pre-run simulation to collect data
    dt = 0.002
    steps = 3000

    all_points = [disk.get_point_positions()]
    for _ in range(steps):
        disk.step(field, dt)
        if _ % 10 == 0:  # Store every 10th frame
            all_points.append(disk.get_point_positions())

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: disk visualization
    field.plot(ax1, xlim=(-2, 2), ylim=(-2, 2))
    ax1.set_aspect('equal')
    ax1.set_title('Spinning Disk in Corrugated Vacuum')

    # Initialize disk visualization
    points = all_points[0]
    disk_plot, = ax1.plot([], [], 'ko-', markersize=4, linewidth=1)
    center_plot, = ax1.plot([], [], 'ro', markersize=10)

    # Right: omega over time
    t_full = np.arange(steps + 1) * dt
    ax2.plot(t_full, disk.omega_history, 'b-', alpha=0.3)
    omega_line, = ax2.plot([], [], 'b-', linewidth=2)
    omega_point, = ax2.plot([], [], 'ro', markersize=8)
    ax2.set_xlim(0, t_full[-1])
    ax2.set_ylim(min(disk.omega_history) * 0.9, max(disk.omega_history) * 1.1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular velocity ω')
    ax2.set_title('Rotational Decay (Radiation Reaction)')
    ax2.grid(True, alpha=0.3)

    def init():
        disk_plot.set_data([], [])
        center_plot.set_data([], [])
        omega_line.set_data([], [])
        omega_point.set_data([], [])
        return disk_plot, center_plot, omega_line, omega_point

    def animate(frame):
        # Disk points (close the loop)
        points = all_points[frame]
        x = np.append(points[:, 0], points[0, 0])
        y = np.append(points[:, 1], points[0, 1])
        disk_plot.set_data(x, y)
        center_plot.set_data([0], [0])

        # Omega trace
        t_idx = frame * 10
        omega_line.set_data(t_full[:t_idx+1], disk.omega_history[:t_idx+1])
        omega_point.set_data([t_full[t_idx]], [disk.omega_history[t_idx]])

        return disk_plot, center_plot, omega_line, omega_point

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(all_points), interval=30, blit=True)

    # Save animation
    print("Saving animation... (this may take a moment)")
    anim.save('experiment4_spinning_disk.gif', writer='pillow', fps=30)
    print("Saved: experiment4_spinning_disk.gif")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CORRUGATED VACUUM SIMULATION")
    print("Exploring Informational Viscosity")
    print("=" * 60)
    print("\nThis simulation tests the hypothesis that rotation experiences")
    print("'friction' in a corrugated vacuum while translation does not.")
    print()

    # Run experiments
    experiment_1_single_particles()
    experiment_2_rotation_vs_translation()
    experiment_3_coupling_strength()

    # Animation (optional - takes longer)
    try:
        experiment_4_animation()
    except Exception as e:
        print(f"Animation skipped: {e}")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print("\nKey observations to look for:")
    print("1. Particles moving along valleys (y) maintain speed")
    print("2. Particles moving across ridges (x) oscillate/slow")
    print("3. Rotating disks lose angular momentum (spin-down)")
    print("4. Translating disks maintain velocity")
    print("5. Higher coupling = faster spin-down (more viscosity)")
    print("\nThis models radiation reaction as 'informational friction'!")
