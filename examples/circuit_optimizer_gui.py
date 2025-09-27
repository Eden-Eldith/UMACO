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
from Umaco13 import create_umaco_solver

GRID_SIZE = 8  # 8x8 board
COMPONENTS = ["A", "B", "C", "D"]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (0, 3)]

# ---------------------------------------------------------------
# Loss function interpreting pheromone matrix as component coords
# ---------------------------------------------------------------

def loss_function(params: np.ndarray) -> float:
    """Loss function for circuit component placement optimization."""
    # params is now a 1D array with 8 values (4 components * 2 coords each)
    coords = []
    for i in range(0, len(params), 2):
        x_val = params[i]
        y_val = params[i + 1]
        # Scale from [0, 2] to grid coordinates
        x = int(np.clip(x_val * (GRID_SIZE - 1) / 2, 0, GRID_SIZE - 1))
        y = int(np.clip(y_val * (GRID_SIZE - 1) / 2, 0, GRID_SIZE - 1))
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
# ---------------------------------------------------------------
# Helper to extract coordinates after optimization
# ---------------------------------------------------------------

def extract_coords_from_params(params: np.ndarray) -> list:
    """Convert flattened parameter vector back into grid coordinates."""
    coords = []
    if params is None or len(params) < len(COMPONENTS) * 2:
        return [(0, 0)] * len(COMPONENTS)

    for i in range(0, len(COMPONENTS) * 2, 2):
        x_param = np.clip(params[i], 0.0, 2.0)
        y_param = np.clip(params[i + 1], 0.0, 2.0)
        x = int(np.round(x_param * (GRID_SIZE - 1) / 2))
        y = int(np.round(y_param * (GRID_SIZE - 1) / 2))
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

        # Create UMACO solver using factory function
        self.solver, self.agents = create_umaco_solver(
            problem_type='CONTINUOUS',
            dim=64,  # Higher resolution for component placement
            max_iter=300,
            problem_dim=8  # 4 components * 2 coordinates each
        )

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

        # Run optimization
        result = self.solver.optimize(self.agents, loss_function)
        best_params = np.asarray(result.best_solution) if result.best_solution is not None else None
        coords = extract_coords_from_params(best_params)

        # Fallback to pheromone interpretation if best solution unavailable
        if best_params is None:
            coords = extract_coords_from_params(result.pheromone_real.flatten()[: len(COMPONENTS) * 2])

        # Calculate final cost using the extracted coordinates
        # Convert coords back to parameter array for loss calculation
        params = []
        for x, y in coords:
            x_param = x * 2 / (GRID_SIZE - 1)
            y_param = y * 2 / (GRID_SIZE - 1)
            params.extend([x_param, y_param])
        total_cost = loss_function(np.array(params))

        self._draw_grid(coords)
        messagebox.showinfo("Optimization Complete", f"Final cost: {total_cost:.2f}")
        self.run_button.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = CircuitOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
