import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig


class RandomWalkGenerator:

    def __init__(
        self,
        cfg: DictConfig,
        p_min,
        p_max,
        p0,
    ):
        self.p0 = p0
        self.p_min = p_min
        self.p_max = p_max
        self.dt = cfg.dt
        self.n_steps = cfg.n_steps
        self.accel_sigma = np.random.uniform(
            cfg.accel_sigma_range[0], cfg.accel_sigma_range[1]
        )
        self.vel_memory = cfg.vel_memory
        self.vmax = cfg.vmax

    def get_trajectory(self):

        pos = np.array(self.p0)
        vel = np.array([0.0, 0.0])
        bbox = (self.p_min[0], self.p_max[0], self.p_min[1], self.p_max[1])

        trajectory = []

        for i in range(self.n_steps):
            next_pos, vel = self.smooth_random_walk_2d(
                pos,
                vel,
                bbox,
            )
            trajectory.append(next_pos)
            pos = next_pos

        return np.array(trajectory)

    def smooth_random_walk_2d(
        self,
        pos,
        vel,
        bbox,
    ):
        """
        pos: np.array([x, y]) : current position
        vel: np.array([vx, vy]) : current velocity
        bbox: (xmin, xmax, ymin, ymax)
        dt: timestep
        accel_sigma: stddev of random acceleration
        vel_memory: 0..1 => 1 = very smooth, 0 = no memory
        vmax: max speed magnitude
        """

        xmin, xmax, ymin, ymax = bbox

        # --------------------------
        # 1. Smooth random acceleration
        # --------------------------
        # OU process for smooth vel changes
        accel = np.random.randn(2) * self.accel_sigma

        # new velocity = (memory)*old + (1-memory)*random_acceleration
        vel = self.vel_memory * vel + (1 - self.vel_memory) * accel

        # --------------------------
        # 2. Limit speed to [0, vmax]
        # --------------------------
        speed = np.linalg.norm(vel)

        if speed > self.vmax:
            vel = vel / speed * self.vmax  # normalize
        # speed < 0 is impossible

        # --------------------------
        # 3. Predict next position
        # --------------------------
        next_pos = pos + vel * self.dt

        # --------------------------
        # 4. Enforce bounding box *before* crossing
        # --------------------------
        # If next step goes outside the box: slide along boundary or bounce
        if next_pos[0] < xmin:
            next_pos[0] = xmin
            vel[0] = abs(vel[0])  # turn inward smoothly
        elif next_pos[0] > xmax:
            next_pos[0] = xmax
            vel[0] = -abs(vel[0])

        if next_pos[1] < ymin:
            next_pos[1] = ymin
            vel[1] = abs(vel[1])
        elif next_pos[1] > ymax:
            next_pos[1] = ymax
            vel[1] = -abs(vel[1])

        return next_pos, vel


def plot_trajectory(points, out_path=None, arrow_scale=1.0, step=1):
    """
    points: (N, 2) array of x,y coordinates
    out_path: file path to save the image (PNG/JPG/etc)
    arrow_scale: scale of arrow lengths
    step: plot arrows every N steps (e.g. step=3 plots every 3rd arrow)
    """

    points = np.asarray(points)
    x = points[:, 0]
    y = points[:, 1]

    # compute velocity vectors (displacements)
    v = np.diff(points, axis=0)

    plt.figure(figsize=(8, 8))

    # trajectory line
    plt.plot(x, y, color="blue", linewidth=2, alpha=0.8)

    # arrows
    for i in range(0, len(v), step):
        px, py = points[i]
        vx, vy = v[i]
        plt.arrow(
            px,
            py,  # starting point
            vx * arrow_scale,  # dx
            vy * arrow_scale,  # dy
            head_width=0.1,
            head_length=0.2,
            fc="red",
            ec="red",
            length_includes_head=True,
        )

    plt.title("2D Trajectory with Step Vectors")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
