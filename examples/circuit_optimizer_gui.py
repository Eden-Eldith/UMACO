#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circuit Design Optimizer Example
================================

This example demonstrates how UMACO9 can be adapted for a very simple
circuit placement problem. Four components must be placed on a small
grid. The objective is to minimize the total Manhattan wiring distance
between connected components while avoiding collisions.

A lightweight Tkinter GUI is provided to run the optimization and
visualize the final layout.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from umaco.Umaco9 import (
    UMACO9, UMACO9Config,
    UniversalEconomy, EconomyConfig,
    UniversalNode, NodeConfig,
    NeuroPheromoneSystem, PheromoneConfig
)

GRID_SIZE = 8  # 8x8 board
COMPONENTS = ["A", "B", "C", "D"]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (0, 3)]

# ---------------------------------------------------------------
# Loss function interpreting pheromone matrix as component coords
# ---------------------------------------------------------------

def loss_function(matrix: np.ndarray) -> float:
    diag = np.diag(matrix)
    coords = []
    for i in range(0, len(COMPONENTS) * 2, 2):
        x_val = diag[i]
        y_val = diag[i + 1]
        x = int(np.clip(x_val * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
        y = int(np.clip(y_val * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
        coords.append((x, y))

    cost = 0.0
    seen = set()
    for c in coords:
        if c in seen:
            cost += 10.0  # heavy penalty for overlapping components
        else:
            seen.add(c)

    for i, j in CONNECTIONS:
        ax, ay = coords[i]
        bx, by = coords[j]
        cost += abs(ax - bx) + abs(ay - by)
    return cost

# ---------------------------------------------------------------
# Helper to extract coordinates after optimization
# ---------------------------------------------------------------

def extract_coords(matrix: np.ndarray):
    diag = np.diag(matrix)
    coords = []
    for i in range(0, len(COMPONENTS) * 2, 2):
        x_val = diag[i]
        y_val = diag[i + 1]
        x = int(np.clip(x_val * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
        y = int(np.clip(y_val * (GRID_SIZE - 1), 0, GRID_SIZE - 1))
        coords.append((x, y))
    return coords

# ---------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------

class CircuitOptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("UMACO Circuit Optimizer")

        self.cell = 40
        self.canvas = tk.Canvas(
            master, width=GRID_SIZE * self.cell, height=GRID_SIZE * self.cell
        )
        self.canvas.pack(padx=10, pady=10)

        self.run_button = tk.Button(master, text="Run Optimization", command=self.run)
        self.run_button.pack(pady=5)

        self._draw_grid()

        # Configure UMACO9 components
        economy_cfg = EconomyConfig(n_agents=8, initial_tokens=200)
        self.economy = UniversalEconomy(economy_cfg)

        pheromone_cfg = PheromoneConfig(n_dim=64, initial_val=0.3)
        self.pheromones = NeuroPheromoneSystem(pheromone_cfg)

        init_vals = np.zeros((64, 64))
        config = UMACO9Config(
            n_dim=64,
            panic_seed=init_vals + 0.1,
            trauma_factor=0.5,
            alpha=0.2,
            beta=0.1,
            rho=0.3,
            max_iter=300,
            quantum_burst_interval=50,
        )
        self.solver = UMACO9(config, self.economy, self.pheromones)
        self.agents = [UniversalNode(i, self.economy, NodeConfig()) for i in range(8)]

    def _draw_grid(self, coords=None):
        self.canvas.delete("all")
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1 = i * self.cell
                y1 = j * self.cell
                x2 = x1 + self.cell
                y2 = y1 + self.cell
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray")
        if coords:
            colors = ["red", "blue", "green", "orange"]
            for idx, (x, y) in enumerate(coords):
                x1 = x * self.cell
                y1 = y * self.cell
                self.canvas.create_rectangle(
                    x1, y1, x1 + self.cell, y1 + self.cell,
                    fill=colors[idx % len(colors)],
                )
                self.canvas.create_text(
                    x1 + self.cell / 2, y1 + self.cell / 2, text=COMPONENTS[idx]
                )

    def run(self):
        self.run_button.config(state=tk.DISABLED)
        pheromone_real, _, _, _ = self.solver.optimize(self.agents, loss_function)
        coords = extract_coords(pheromone_real)
        self._draw_grid(coords)
        total_cost = loss_function(pheromone_real)
        messagebox.showinfo("Optimization Complete", f"Final cost: {total_cost:.2f}")
        self.run_button.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = CircuitOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
