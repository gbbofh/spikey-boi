#!/usr/bin/env python3

import pickle

import numpy as np
import pygame

import util
import network

class Graph3D:
    def __init__(self, net: network.Network):
        self.net = net
        self.num_neurons = net.num_neurons

        # Define grid dimensions for a 4x4 grid
        rows = 4
        cols = 4

        # Generate grid positions for 16 nodes (indices 0 to 15)
        x_positions = np.linspace(-1, 1, cols)  # 4 columns
        y_positions = np.linspace(-1, 1, rows)  # 4 rows
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)

        # Flatten the grid arrays
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        # Initialize node positions for all neurons
        self.node_x = np.random.uniform(-0.8, 0.8, self.num_neurons)
        self.node_y = np.random.uniform(-0.8, 0.8, self.num_neurons)
        self.node_z = np.random.uniform(-0.8, 0.8, self.num_neurons)

        # Assign positions to nodes 0 to 15 to form a 4x4 grid
        self.node_x[:16] = grid_x[:16]
        self.node_y[:16] = grid_y[:16]
        self.node_z[:16] = -2  # Place the grid at z = -1 (or adjust as needed)

        # Optional: Assign specific positions to other nodes as needed
        # For example, positions for motor output neurons
        self.node_x[16] = -0.5  # Adjust position for node 16
        self.node_y[16] = 0.2
        self.node_z[16] = 2

        self.node_x[17] = 0.5  # Adjust position for node 17
        self.node_y[17] = 0.2
        self.node_z[17] = 2

        self.spike_trace = np.zeros_like(net.spikes, dtype=np.float64)

        # Node base colors (set base colors for excitatory and inhibitory neurons)
        self.node_r = np.zeros(self.num_neurons)
        self.node_g = np.zeros(self.num_neurons)
        self.node_b = np.zeros(self.num_neurons)

        self.node_r[net.num_exc:] = 255  # Inhibitory neurons in red
        self.node_g[:net.num_exc] = 255  # Excitatory neurons in green
        self.node_b[:] = 100

        # Input
        self.node_r[0:16] = 200
        self.node_g[0:16] = 0
        self.node_b[0:16] = 200

        # Output
        self.node_r[16:18] = 100
        self.node_g[16:18] = 100
        self.node_b[16:18] = 255

        self.alpha = 0.9999

        self.edges = [(i, j, weight) for i, row in enumerate(net.w) for j, weight in enumerate(row) if weight > 0]
        self.projected_pos = {}

    def get_node_colors(self):
        """Adjusts node colors based on their distance from the view."""
        max_distance = 2  # Define the maximum distance for darkening effect
        min_brightness = 0.2  # Minimum brightness factor (20%)
        brightness_factors = np.clip(1 - (self.node_z + max_distance) / (2 * max_distance), min_brightness, 1)

        # Apply brightness factors to the base colors
        node_colors = np.stack((
            (self.node_r * brightness_factors).astype(int),
            (self.node_g * brightness_factors).astype(int),
            (self.node_b * brightness_factors).astype(int)
        ), axis=-1)
        
        return node_colors

    def project(self, x, y, z, width, height, fov=500, viewer_distance=4):
        """ Projects 3D coordinates onto a 2D screen """
        factor = fov / (viewer_distance + z)
        x = x * factor + width / 2
        y = -y * factor + height / 2
        return int(x), int(y)

    def rotate_y(self, angle):
        """ Rotates the 3D coordinates of the nodes around the y-axis by the given angle """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        for i in range(self.num_neurons):
            x = self.node_x[i]
            z = self.node_z[i]
            self.node_x[i] = x * cos_angle - z * sin_angle
            self.node_z[i] = x * sin_angle + z * cos_angle

    def update(self):
        self.edges = [(i, j, weight) for i, row in enumerate(net.w) for j, weight in enumerate(row) if weight > 0]
        self.spike_trace = self.alpha * net.spikes + (1 - self.alpha) * self.spike_trace

        self.node_r[:] = 0
        self.node_g[:] = 0
        self.node_b[:] = 0

        # Default colors
        self.node_r[net.num_exc:] = 255  # Inhibitory neurons in red
        self.node_g[:net.num_exc] = 255  # Excitatory neurons in green
        self.node_b[:] = 100

        # Input
        self.node_r[0:16] = 200
        self.node_g[0:16] = 0
        self.node_b[0:16] = 200

        # Output
        self.node_r[16:18] = 100
        self.node_g[16:18] = 100
        self.node_b[16:18] = 255

        # syn = net.I_syn

        # neg = syn < 0
        # pos = syn > 0

        # self.node_r[neg] = 255
        # self.node_b[neg] = 0
        # self.node_r[pos] = 0
        # self.node_b[pos] = 255

        # mask = net.spikes > 0
        # self.spike_trace[mask] = 1
        # self.spike_trace *= np.exp(-25 / net.params.dt)

        mask = self.spike_trace > 0

        self.node_r[mask] = 255
        self.node_g[mask] = 255
        self.node_b[mask] = 0

    def draw(self, screen, width, height):
        """ Draws the network nodes and edges on the screen """
        # Calculate the colors for each node based on distance
        node_colors = self.get_node_colors()

        # Calculate the colors for each node based on distance
        node_colors = self.get_node_colors()

        # Calculate depth for each node based on its z-coordinate
        node_depths = [(i, self.node_z[i]) for i in range(self.num_neurons)]

        # Calculate depth for each edge as the average z-coordinate of its endpoints
        edge_depths = [(i, j, (self.node_z[i] + self.node_z[j]) / 2, w) for (i, j, w) in self.edges]

        # Sort nodes by depth (farthest first, closest last)
        node_depths.sort(key=lambda x: x[1], reverse=True)

        # Sort edges by depth (farthest first, closest last)
        edge_depths.sort(key=lambda x: x[2], reverse=True)

        # Draw edges
        # for i, j, weight in self.edges:
        for i, j, _, weight in edge_depths:
            # Project the 3D positions to 2D
            x1, y1 = self.project(self.node_x[i], self.node_y[i], self.node_z[i], width, height)
            x2, y2 = self.project(self.node_x[j], self.node_y[j], self.node_z[j], width, height)
            # color = (200, 200, 200)  # Edge color
            color = node_colors[i] // 2
            pygame.draw.line(screen, color, (x1, y1), (x2, y2), max(1, int(weight * 5)))  # Adjust thickness

        # Draw nodes with adjusted colors
        # for i in range(self.num_neurons):
        for i, _ in node_depths:
            x, y = self.project(self.node_x[i], self.node_y[i], self.node_z[i], width, height)
            color = node_colors[i]
            pygame.draw.circle(screen, color, (x, y), 5)  # Adjust radius as needed
            self.projected_pos[i] = (x,y)

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("3D Neural Network Visualization")

# Load network and create Graph3D instance
net = network.Network(50)  # Initialize your network class here
graph = Graph3D(net)

with open('state', 'rb') as fp:
    try:
        while True:
            obj = pickle.load(fp)
            if obj.__class__.__name__ == 'Network':
                net = obj
                break
    except:
        pass

# Main loop
running = True
click_radius = 10

net.I_inj[:] = 0
net.I_syn[:] = 0
net.I_total[:] = 0

while running:
    screen.fill((0, 0, 0))  # Clear screen

    angle = 0  # Initialize rotation angle
    rotation_speed = 0.002  # Set the speed of rotation

    # net.I_inj[:] = np.random.normal(0, net.I_inj.shape)

    net.update()
    net.I_inj *= np.exp(-net.params.dt / 50)
    # print(net.I_inj)

    # Rotate and draw the graph
    graph.update()
    graph.rotate_y(rotation_speed)
    graph.draw(screen, width, height)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x,y = event.pos

            for i, (nx,ny) in graph.projected_pos.items():
                if (x - nx) ** 2 + (y - ny) ** 2 <= click_radius ** 2:
                    net.I_inj[i] += 1

    # Update display
    pygame.display.flip()

pygame.quit()

