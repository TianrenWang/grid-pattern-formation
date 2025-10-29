# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt


class TrajectoryGenerator(object):
    def __init__(self, options, place_cells):
        self.options = options
        self.place_cells = place_cells

    def plot_trajectory(self, traj, box_width, box_height, idx=0, step=2):
        """
        Visualize one trajectory from traj dict.

        Args:
            traj: dictionary containing trajectory info
            box_width, box_height: dimensions of the environment
            idx: which trajectory to plot from the batch (default: 0)
            step: plot an arrow every 'step' frames
        """
        # Extract trajectory for one rat
        x = traj["target_x"][idx]  # shape (samples,)
        y = traj["target_y"][idx]
        hd = traj["target_hd"][idx]  # head directions in radians

        # Also add starting point
        x0 = traj["init_x"][idx, 0]
        y0 = traj["init_y"][idx, 0]
        hd0 = traj["init_hd"][idx, 0]

        x = np.concatenate([[x0], x])
        y = np.concatenate([[y0], y])
        hd = np.concatenate([[hd0], hd])

        # Plot trajectory
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, "-o", markersize=2, label="trajectory")

        # Add arrows for head direction
        for t in range(0, len(x), step):
            dx = 0.1 * np.cos(hd[t])
            dy = 0.1 * np.sin(hd[t])
            plt.arrow(
                x[t], y[t], dx, dy, head_width=0.05, head_length=0.08, fc="r", ec="r"
            )

        # Draw box boundaries
        plt.axhline(y=0, color="k")
        plt.axhline(y=box_height, color="k")
        plt.axvline(x=0, color="k")
        plt.axvline(x=box_width, color="k")

        plt.xlim([-0.2, box_width + 0.2])
        plt.ylim([-0.2, box_height + 0.2])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("x position (m)")
        plt.ylabel("y position (m)")
        plt.title(f"Trajectory {idx}")
        plt.legend()
        plt.show()

    def avoid_wall(self, position, hd, box_width, box_height):
        """
        Compute distance and angle to nearest wall
        """
        x = position[:, 0]
        y = position[:, 1]
        dists = [
            box_width - x,
            box_height - y,
            x,
            y,
        ]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (
            np.pi / 2 - np.abs(a_wall[is_near_wall])
        )

        return is_near_wall, turn_angle

    def generate_trajectory(self, box_width, box_height, batch_size):
        """Generate a random walk in a rectangular box"""
        samples = self.options.sequence_length
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples + 2, 2])
        head_dir = np.zeros([batch_size, samples + 2])
        position[:, 0, 0] = np.random.uniform(0, box_width, batch_size)
        position[:, 0, 1] = np.random.uniform(0, box_height, batch_size)
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        velocity = np.zeros([batch_size, samples + 2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])
        random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
        v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

        for t in range(samples + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            if not self.options.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(
                    position[:, t], head_dir[:, t], box_width, box_height
                )
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack(
                [np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1
            )
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        # Periodic boundaries
        if self.options.periodic:
            position[:, :, 0] = (
                np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
            )
            position[:, :, 1] = (
                np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2
            )

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        traj = {}
        # Input variables
        traj["init_hd"] = head_dir[:, 0, None]
        traj["init_x"] = position[:, 1, 0, None]
        traj["init_y"] = position[:, 1, 1, None]

        traj["ego_v"] = velocity[:, 1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        traj["phi_x"], traj["phi_y"] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

        # Target variables
        traj["target_hd"] = head_dir[:, 1:-1]
        traj["target_x"] = position[:, 2:, 0]
        traj["target_y"] = position[:, 2:, 1]

        # for i in range(5):
        #     self.plot_trajectory(traj, box_width, box_height, i)
        # raise Exception("dog")

        return traj

    def get_generator(self, batch_size=None, box_width=None, box_height=None):
        """
        Returns a generator that yields batches of trajectories
        """
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        while True:
            traj = self.generate_trajectory(box_width, box_height, batch_size)

            v = np.stack(
                [
                    traj["ego_v"] * np.cos(traj["target_hd"]),
                    traj["ego_v"] * np.sin(traj["target_hd"]),
                ],
                axis=-1,
            )
            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

            pos = np.stack([traj["target_x"], traj["target_y"]], axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
            # Put on GPU if GPU is available
            pos = pos.to(self.options.device)
            place_outputs = self.place_cells.get_activation(pos)

            init_pos = np.stack([traj["init_x"], traj["init_y"]], axis=-1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to(self.options.device)
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            v = v.to(self.options.device)
            inputs = (v, init_actv)

            yield (inputs, place_outputs, pos)

    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):
        """For testing performance, returns a batch of smample trajectories"""
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        traj = self.generate_trajectory(box_width, box_height, batch_size)

        v = np.stack(
            [
                traj["ego_v"] * np.cos(traj["target_hd"]),
                traj["ego_v"] * np.sin(traj["target_hd"]),
            ],
            axis=-1,
        )
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

        pos = np.stack([traj["target_x"], traj["target_y"]], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.options.device)
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj["init_x"], traj["init_y"]], axis=-1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.options.device)
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        v = v.to(self.options.device)
        inputs = (v, init_actv)

        return (inputs, pos, place_outputs)
