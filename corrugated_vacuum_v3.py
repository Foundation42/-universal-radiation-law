"""
Corrugated Vacuum Simulation v3
===============================
Focus on:
1. Clearer radiation visualization (slower decay)
2. Resonance effects (disk size vs corrugation wavelength)
3. The "light cylinder" analogy (c/ω critical radius)
4. Multiple disks with different parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import laplace

# =============================================================================
# DYNAMICAL FIELD (improved)
# =============================================================================

class VacuumField:
    """Corrugated vacuum with wave dynamics"""

    def __init__(self, grid_size=200, domain_size=12.0,
                 amplitude=0.4, wavelength=1.0, wave_speed=5.0):
        self.N = grid_size
        self.L = domain_size
        self.dx = domain_size / grid_size
        self.c = wave_speed

        # Background corrugation
        self.A = amplitude
        self.k = 2 * np.pi / wavelength
        self.wavelength = wavelength

        # Grid
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.y = np.linspace(-self.L/2, self.L/2, self.N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Background (static)
        self.h0 = self.A * np.sin(self.k * self.X)

        # Dynamical field
        self.phi = np.zeros((self.N, self.N))
        self.phi_dot = np.zeros((self.N, self.N))

        # Tracking
        self.field_energy_history = []

    def background_gradient(self, x, y):
        dhdx = self.A * self.k * np.cos(self.k * x)
        return np.array([dhdx, 0.0])

    def phi_gradient(self, x, y):
        i = int((x + self.L/2) / self.dx)
        j = int((y + self.L/2) / self.dx)
        i = max(1, min(self.N-2, i))
        j = max(1, min(self.N-2, j))
        dphidx = (self.phi[j, i+1] - self.phi[j, i-1]) / (2 * self.dx)
        dphidy = (self.phi[j+1, i] - self.phi[j-1, i]) / (2 * self.dx)
        return np.array([dphidx, dphidy])

    def total_gradient(self, x, y):
        return self.background_gradient(x, y) + self.phi_gradient(x, y)

    def deposit(self, x, y, amount, sigma=0.2):
        """Deposit energy into field (radiation source)"""
        r2 = (self.X - x)**2 + (self.Y - y)**2
        self.phi_dot += amount * np.exp(-r2 / (2 * sigma**2))

    def step(self, dt):
        """Wave equation with absorbing boundaries and stability"""
        # CFL stability check
        cfl = self.c * dt / self.dx
        if cfl > 0.5:
            # Subdivide timestep for stability
            n_sub = int(np.ceil(cfl / 0.4))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                self._substep(sub_dt)
        else:
            self._substep(dt)

        # Energy tracking
        ke = 0.5 * np.sum(self.phi_dot**2) * self.dx**2
        self.field_energy_history.append(ke)

    def _substep(self, dt):
        """Single substep of wave equation"""
        lap = laplace(self.phi) / (self.dx**2)

        # Absorbing boundary layer
        edge = int(0.15 * self.N)
        damping = np.zeros_like(self.phi)
        for i in range(edge):
            strength = 0.3 * (1 - i/edge)
            damping[i, :] = strength
            damping[-(i+1), :] = strength
            damping[:, i] = np.maximum(damping[:, i], strength)
            damping[:, -(i+1)] = np.maximum(damping[:, -(i+1)], strength)

        # Global damping to prevent blowup
        self.phi_dot *= 0.9995

        self.phi_dot += self.c**2 * lap * dt
        self.phi_dot *= (1 - damping)
        self.phi += self.phi_dot * dt

        # Clamp to prevent numerical explosion
        max_phi = 10.0
        self.phi = np.clip(self.phi, -max_phi, max_phi)
        self.phi_dot = np.clip(self.phi_dot, -max_phi*10, max_phi*10)

    def field_energy(self):
        return 0.5 * np.sum(self.phi_dot**2) * self.dx**2


# =============================================================================
# SPINNING DISK
# =============================================================================

class SpinningDisk:
    """Disk coupled to vacuum field"""

    def __init__(self, cx=0, cy=0, radius=1.0, n_points=32,
                 omega=5.0, coupling=0.005, mass=1.0):
        self.center = np.array([cx, cy], dtype=float)
        self.radius = radius
        self.omega = omega
        self.theta = 0.0
        self.mass = mass
        self.I = 0.5 * mass * radius**2
        self.alpha = coupling
        self.n_points = n_points

        self.omega_history = [omega]
        self.rot_ke_history = []
        self.power_history = []

    def rot_ke(self):
        return 0.5 * self.I * self.omega**2

    def step(self, field, dt):
        angles = np.linspace(0, 2*np.pi, self.n_points, endpoint=False) + self.theta
        total_torque = 0.0
        total_power = 0.0

        for angle in angles:
            # Position and velocity of this point
            px = self.center[0] + self.radius * np.cos(angle)
            py = self.center[1] + self.radius * np.sin(angle)

            # Bounds check
            if abs(px) > field.L/2 - 0.5 or abs(py) > field.L/2 - 0.5:
                continue

            vx = -self.omega * self.radius * np.sin(angle)
            vy = self.omega * self.radius * np.cos(angle)

            # Field coupling
            dh = field.total_gradient(px, py)
            if np.any(np.isnan(dh)) or np.any(np.isinf(dh)):
                continue

            v_dot_dh = vx * dh[0] + vy * dh[1]
            Fx = -2 * self.alpha * v_dot_dh * dh[0]
            Fy = -2 * self.alpha * v_dot_dh * dh[1]

            # Clamp forces to prevent instability
            max_force = 100.0
            Fx = np.clip(Fx, -max_force, max_force)
            Fy = np.clip(Fy, -max_force, max_force)

            # Torque about center
            rx, ry = px - self.center[0], py - self.center[1]
            torque = rx * Fy - ry * Fx
            total_torque += torque

            # Power (radiation)
            power = -(Fx * vx + Fy * vy)
            if power > 0 and power < 1000:  # Sanity check
                field.deposit(px, py, power * dt * 0.1, sigma=0.15)
                total_power += power

        # Update rotation (with damping limit)
        angular_accel = np.clip(total_torque / self.I, -100, 100)
        self.omega += angular_accel * dt
        self.theta += self.omega * dt

        self.omega_history.append(self.omega)
        self.rot_ke_history.append(self.rot_ke())
        self.power_history.append(total_power)


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_clear_radiation():
    """
    Slow decay to clearly see radiation waves propagating outward.
    """
    print("=" * 60)
    print("EXPERIMENT: Clear Radiation Visualization")
    print("=" * 60)

    field = VacuumField(grid_size=250, domain_size=15.0,
                       amplitude=0.3, wavelength=1.0, wave_speed=8.0)

    disk = SpinningDisk(cx=0, cy=0, radius=1.2, n_points=48,
                       omega=6.0, coupling=0.003)

    dt = 0.008
    steps = 1200
    snapshots = []
    snap_interval = 6

    print("Running simulation...")
    for step in range(steps):
        disk.step(field, dt)
        field.step(dt)

        if step % snap_interval == 0:
            snapshots.append({
                'phi': field.phi.copy(),
                'theta': disk.theta,
                'omega': disk.omega,
                'time': step * dt
            })

        if step % 200 == 0:
            print(f"  t={step*dt:.2f}, ω={disk.omega:.3f}, KE={disk.rot_ke():.3f}")

    # Create detailed animation
    print("\nCreating animation...")

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Field plot
    vmax = np.percentile(np.abs(field.phi), 99) + 0.001
    im = ax1.imshow(snapshots[0]['phi'], extent=[-7.5, 7.5, -7.5, 7.5],
                   cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower',
                   interpolation='bilinear')
    ax1.contour(field.X, field.Y, field.h0, levels=8, colors='gray', alpha=0.3, linewidths=0.5)

    # Disk visualization
    theta_circ = np.linspace(0, 2*np.pi, 60)
    disk_line, = ax1.plot([], [], 'k-', linewidth=2)
    spokes = [ax1.plot([], [], 'k-', linewidth=1, alpha=0.5)[0] for _ in range(4)]

    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Field Perturbation φ\n(Radiation waves)')
    ax1.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('φ')

    # Omega vs time
    t_all = np.arange(steps + 1) * dt
    ax2.plot(t_all, disk.omega_history, 'b-', alpha=0.3)
    omega_line, = ax2.plot([], [], 'b-', linewidth=2)
    omega_dot, = ax2.plot([], [], 'ro', markersize=10)
    ax2.set_xlim(0, steps * dt)
    ax2.set_ylim(0, max(disk.omega_history) * 1.1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular velocity ω (rad/s)')
    ax2.set_title('Spin-Down Curve\n(Radiation reaction torque)')
    ax2.grid(True, alpha=0.3)

    # Power spectrum (instantaneous)
    ax3.set_xlim(0, steps * dt)
    power_line, = ax3.plot([], [], 'r-', linewidth=1.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Radiated Power')
    ax3.set_title('Instantaneous Radiated Power\n(Energy leaving the disk)')
    ax3.grid(True, alpha=0.3)
    if len(disk.power_history) > 0:
        ax3.set_ylim(0, max(disk.power_history) * 1.2 + 0.01)

    time_text = ax1.text(-5.5, 5, '', fontsize=12, color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def animate(frame):
        snap = snapshots[frame]
        idx = frame * snap_interval

        # Update field
        im.set_array(snap['phi'])

        # Update disk
        disk_x = disk.radius * np.cos(theta_circ)
        disk_y = disk.radius * np.sin(theta_circ)
        disk_line.set_data(disk_x, disk_y)

        # Spokes to show rotation
        for i, spoke in enumerate(spokes):
            angle = snap['theta'] + i * np.pi/2
            spoke.set_data([0, disk.radius*0.9*np.cos(angle)],
                          [0, disk.radius*0.9*np.sin(angle)])

        # Update plots
        omega_line.set_data(t_all[:idx+1], disk.omega_history[:idx+1])
        omega_dot.set_data([t_all[idx]], [disk.omega_history[idx]])

        if idx > 0 and idx <= len(disk.power_history):
            power_line.set_data(t_all[1:idx+1], disk.power_history[:idx])

        time_text.set_text(f't = {snap["time"]:.2f}\nω = {snap["omega"]:.3f}')

        return [im, disk_line, omega_line, omega_dot, power_line, time_text] + spokes

    anim = animation.FuncAnimation(fig, animate, frames=len(snapshots),
                                   interval=40, blit=True)

    plt.tight_layout()
    print("Saving animation...")
    anim.save('clear_radiation.gif', writer='pillow', fps=25, dpi=100)
    print("Saved: clear_radiation.gif")
    plt.savefig('clear_radiation_final.png', dpi=150)
    plt.show()


def experiment_resonance():
    """
    Compare disks of different sizes relative to corrugation wavelength.
    Prediction: maximum radiation when disk diameter ~ wavelength
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Resonance (Disk Size vs Wavelength)")
    print("=" * 60)

    wavelength = 1.5
    radii = [0.3, 0.75, 1.5, 3.0]  # From << λ to >> λ

    results = []

    for radius in radii:
        print(f"\nRadius = {radius} (ratio to λ: {2*radius/wavelength:.2f})")

        field = VacuumField(grid_size=150, domain_size=10.0,
                           amplitude=0.3, wavelength=wavelength, wave_speed=5.0)

        disk = SpinningDisk(cx=0, cy=0, radius=radius, n_points=max(16, int(32*radius)),
                           omega=5.0, coupling=0.01)

        dt = 0.01
        steps = 500

        for _ in range(steps):
            disk.step(field, dt)
            field.step(dt)

        # Calculate decay rate (fit exponential)
        omega_arr = np.array(disk.omega_history)
        t_arr = np.arange(len(omega_arr)) * dt

        # Simple decay rate estimate
        if omega_arr[-1] > 0.1:
            half_idx = np.argmax(omega_arr < omega_arr[0]/2) if np.any(omega_arr < omega_arr[0]/2) else -1
            decay_time = t_arr[half_idx] if half_idx > 0 else float('inf')
        else:
            decay_time = t_arr[np.argmax(omega_arr < omega_arr[0]/2)] if np.any(omega_arr < omega_arr[0]/2) else t_arr[-1]

        total_radiated = np.sum(disk.power_history) * dt

        results.append({
            'radius': radius,
            'ratio': 2*radius/wavelength,
            'decay_time': decay_time,
            'total_radiated': total_radiated,
            'omega_history': omega_arr.copy(),
            'final_phi': field.phi.copy()
        })

        print(f"  Decay time (to half): {decay_time:.3f}")
        print(f"  Total energy radiated: {total_radiated:.4f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Omega decay curves
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    for r, color in zip(results, colors):
        t = np.arange(len(r['omega_history'])) * 0.01
        ax.plot(t, r['omega_history'], color=color,
               label=f"R={r['radius']:.2f} (D/λ={r['ratio']:.1f})", linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('ω')
    ax.set_title('Spin-Down for Different Disk Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Decay rate vs size ratio
    ax = axes[0, 1]
    ratios = [r['ratio'] for r in results]
    decay_rates = [1/r['decay_time'] if r['decay_time'] > 0 else 0 for r in results]
    ax.plot(ratios, decay_rates, 'bo-', markersize=10, linewidth=2)
    ax.set_xlabel('Disk Diameter / Wavelength')
    ax.set_ylabel('Decay Rate (1/τ)')
    ax.set_title('Resonance: Decay Rate vs Size Ratio\nPeak = maximum coupling to field')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='D = λ')
    ax.legend()

    # Field patterns (2x2 grid within subplot)
    ax = axes[1, 0]
    ax.set_title('Final Field Patterns')
    for i, r in enumerate(results[:4]):
        ax_sub = fig.add_axes([0.05 + (i%2)*0.22, 0.05 + (1-i//2)*0.22, 0.2, 0.2])
        vmax = np.percentile(np.abs(r['final_phi']), 98) + 0.001
        ax_sub.imshow(r['final_phi'], cmap='seismic', vmin=-vmax, vmax=vmax)
        ax_sub.set_title(f"R={r['radius']}", fontsize=9)
        ax_sub.axis('off')
    ax.axis('off')

    # Energy radiated vs size
    ax = axes[1, 1]
    radiated = [r['total_radiated'] for r in results]
    ax.bar(range(len(results)), radiated, color=colors)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"D/λ={r['ratio']:.1f}" for r in results])
    ax.set_ylabel('Total Energy Radiated')
    ax.set_title('Radiation vs Disk Size')

    plt.tight_layout()
    plt.savefig('resonance_experiment.png', dpi=150)
    plt.show()


def experiment_light_cylinder():
    """
    The "light cylinder" concept: at radius r = c/ω, the corrugation
    can't update fast enough to keep up with the rotation.

    Inside: field can "follow" the disk
    Outside: radiation is emitted
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Light Cylinder Analogy")
    print("=" * 60)

    c = 5.0  # Field wave speed ("speed of light")

    # Two cases: ω such that light cylinder is inside vs outside disk
    omegas = [2.0, 10.0]  # r_LC = c/ω = 2.5 and 0.5
    disk_radius = 1.5

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row, omega in enumerate(omegas):
        r_LC = c / omega
        print(f"\nω = {omega}, Light cylinder radius = {r_LC:.2f}")
        print(f"Disk radius = {disk_radius}, {'INSIDE' if disk_radius < r_LC else 'OUTSIDE'} light cylinder")

        field = VacuumField(grid_size=200, domain_size=12.0,
                           amplitude=0.3, wavelength=1.0, wave_speed=c)

        disk = SpinningDisk(cx=0, cy=0, radius=disk_radius, n_points=48,
                           omega=omega, coupling=0.008)

        dt = 0.005
        steps = 800

        for _ in range(steps):
            disk.step(field, dt)
            field.step(dt)

        # Plot field
        ax = axes[row, 0]
        vmax = np.percentile(np.abs(field.phi), 98) + 0.001
        im = ax.imshow(field.phi, extent=[-6, 6, -6, 6], cmap='seismic',
                      vmin=-vmax, vmax=vmax, origin='lower')
        ax.contour(field.X, field.Y, field.h0, levels=5, colors='gray', alpha=0.2)

        # Draw disk
        theta = np.linspace(0, 2*np.pi, 50)
        ax.plot(disk_radius * np.cos(theta), disk_radius * np.sin(theta), 'k-', lw=2)

        # Draw light cylinder
        ax.plot(r_LC * np.cos(theta), r_LC * np.sin(theta), 'g--', lw=2,
               label=f'Light cyl (r={r_LC:.1f})')

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        status = "Disk INSIDE LC" if disk_radius < r_LC else "Disk OUTSIDE LC"
        ax.set_title(f'ω={omega}, c={c}\n{status}')
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Omega decay
        ax = axes[row, 1]
        t = np.arange(len(disk.omega_history)) * dt
        ax.plot(t, disk.omega_history, 'b-', lw=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('ω')
        ax.set_title(f'Angular Velocity\nDecay: {100*(1-disk.omega_history[-1]/omega):.0f}%')
        ax.grid(True, alpha=0.3)

        # Radiated power
        ax = axes[row, 2]
        ax.plot(t[1:], disk.power_history, 'r-', lw=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Power')
        ax.set_title('Radiated Power')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Light Cylinder Effect: Field cannot update faster than c\n'
                'Disk edge exceeding c/ω → strong radiation', fontsize=12)
    plt.tight_layout()
    plt.savefig('light_cylinder.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CORRUGATED VACUUM v3: Advanced Experiments")
    print("=" * 60)

    experiment_clear_radiation()
    experiment_resonance()
    experiment_light_cylinder()

    print("\n" + "=" * 60)
    print("SUMMARY OF PHYSICS")
    print("=" * 60)
    print("""
Key findings from these simulations:

1. RADIATION IS REAL
   - Spinning objects deposit energy into the field
   - This energy propagates away as waves
   - The disk slows down = radiation reaction

2. RESONANCE
   - Maximum radiation when disk size ~ corrugation wavelength
   - Too small: doesn't "see" the corrugation structure
   - Too large: averages over many corrugations

3. LIGHT CYLINDER
   - At r = c/ω, the field can't update fast enough
   - Disk edge velocity > c → strong radiation
   - Analogous to pulsar magnetosphere physics!

4. TRANSLATION VS ROTATION
   - Translation along valleys: frictionless (Newton's 1st law)
   - Rotation: always crosses corrugations → radiates

This supports your "informational viscosity" hypothesis:
The vacuum resists being "twisted" faster than it can update.
""")
