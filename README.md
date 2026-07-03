# NeuroSwarm

A 2D simulation of learned, emergent predator-prey behavior using neural-network-driven agents and genetic evolution algorithms (neuroevolution). 

Rather than using hardcoded flocking or boids rules, **NeuroSwarm** prey agents (fish) independently learn behaviors like schooling, predator evasion, and border avoidance through generations of mutation and selection.

---

## 📺 Simulation Demo

*A GIF or video demonstrating emergent schooling behavior and predator evasion should be placed here.*

*(To record a demo, run the simulation, press `G` to show the fitness graph, and capture a 10-second capture of the window).*

---

## 🌟 Key Features

- **Autonomous Neural Controllers**: Every fish is steered by an independent feed-forward neural network ($5$ inputs $\rightarrow$ $8$ hidden nodes $\rightarrow$ $2$ output steering/speed forces) using pure NumPy matrix multiplication.
- **Dynamic Vision & Wall Raycasts**: Fish read physical sensors (toroidal nearest-neighbor vector, proximity to screen borders, and relative vector to the predator).
- **Emergent Schooling & School Flocking**: Anti-predator behaviors (grouping, splitting, darting) emerge naturally without pre-programmed instructions.
- **Rule-Based Hunter Agent**: A predator chases the nearest prey using intercept math, accelerating in lunges.
- **Headless Fast Training Accelerator**: Disable rendering and clock caps (`T` key) to cycle generations up to **30x–60x faster** than real time.
- **Live Vector Plotting**: Overlay real-time charts (`G` key) graphing best and average fitness per generation with dynamic scaling.
- **Keyboard Shortcuts Legend Dashboard**: On-screen menu (`H` key) to toggle and explore commands.
- **Fullscreen Resolution Rescaling**: Seamlessly scale all coordinates, wrap bounds, and graphics panels on the fly (`F` key) without resetting the population.
- **Genome Save & Load**: Save weights (`S` key) to `best_genome.json` and inject them as a visual **Golden Champion** (`L` key) in the next generation.

---

## 🛠️ Requirements

- **Python**: Version `3.8` or higher
- **Dependencies**: 
  - `pygame` (version `2.5.0` or higher)
  - `numpy` (version `1.24.0` or higher)

---

## 🚀 Installation & Launch

1. **Clone the repository**:
   ```bash
   git clone https://github.com/parth-pawarr/Finding-Nemo.git
   cd Finding-Nemo
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the simulation**:
   ```bash
   python main.py
   ```

---

## 🎮 Keyboard Controls Mappings

| Control Shortcut | Action Description | Visual Display Outcome |
| :--- | :--- | :--- |
| **`ESC`** | Quit | Closes window and terminates process |
| **`D`** | Toggle Debug Overlay | Shows hover target vision circle, neighbor vector, wall raycast, and activations |
| **`G`** | Toggle Fitness Graph | Shows/hides live progression charts (bottom-right) |
| **`H`** | Toggle Controls Panel | Shows/hides Keyboard Controls legend (top-right) |
| **`R`** | Toggle Simple Rendering | Hides panels, sets fish to uniform blue for minimal clutter |
| **`F`** | Toggle Fullscreen | Scales coordinates, wrap bounds, and panels to monitor dimensions |
| **`T`** | Toggle Fast Training | Suspends draws, uncaps framerates (updates progress HUD once/sec) |
| **`S`** | Save Best Genome | Serializes flat array weights of the all-time champion to `best_genome.json` |
| **`L`** | Load Best Genome | Loads weight coefficients from disk to inject as **Golden Champion** |
| **`E`** | Export Stats CSV | Exports generation history table to `generation_history.csv` |

---

## ⚙️ How it Works (Neuroevolution)

NeuroSwarm implements a reinforcement-like genetic training cycle:

```
    [ Sense Environment ]  --> vision sensors, wall raycast, predator vectors
             │
             ▼
      [ Neural Brain ]     --> 5 -> 8 -> 2 FF-NN activation (NumPy tanh)
             │
             ▼
    [ Physical Action ]    --> Steering force & acceleration changes (physics clamp)
             │
             ▼
   [ Reward Calculation ]  --> Fitness += Survival time + Speed - Wall proximity penalty
             │
             ▼
    [ Natural Selection ]  --> Elite preservation (15%) + Tournament selection (size 5)
             │
             ▼
  [ Crossover & Mutation ] --> Uniform weight combination + Gaussian mutation noise
             │
             ▼
     [ Next Generation ]   --> Spawns 100 new fish with inherited weights
```

---

## 📂 Project Architecture

```
Finding-Nemo/
│
├── config.py            # Global constants (physics constraints, display sizes, GA params)
├── utils.py             # Math helpers (toroidal coordinate translation) and color generator
├── neural_network.py    # NumPy feed-forward NeuralNetwork weight controller
├── entities.py          # Coordinate updates and sensors for Agent (Base), Fish, and Predator
├── evolution.py         # Genetic Operators (Selection, Uniform Crossover, Gaussian Mutation)
├── analytics.py         # Overlay GUI drawers (HUD, graph, controls panel) and CSV exporter
├── persistence.py       # JSON serialization (load/save best genome weights)
├── main.py              # Central simulator entry point, clock ticker, and keyboard listener
│
├── requirements.txt     # List of library dependencies
└── .gitignore           # Ignores byte caches, local environment folders, and data logs
```

---

## 🗺️ Roadmap & Status (Prototype 1)

This repository holds the completion of **Prototype 1** (Stages 1–8).
- **Core Sim**: Evolved neural fish behaving, responding, and surviving against a rule-based hunter.
- **Implemented Optimization**: Fast headless training loops to decrease evaluation wait times.
- **Roadmap / Out of Scope**:
  - Schooling reward: Left out by design to ensure group cohesion emerges purely as a survival trait, rather than a direct incentive.
  - Spatial partitioning: Deferred for later phases when scaling populations past $100$ agents.

---

## 📄 License

This project is licensed under the terms of the MIT License.
