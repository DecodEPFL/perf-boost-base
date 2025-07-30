# Performance Boosting Controllers

A PyTorch implementation of state feedback performance-boosting controllers based on the paper "Learning to Boost the Performance of Stable Nonlinear Systems".

**Developers:** Mahrokh Ghoddousi Boroujeni, Clara Lucía Galimberti

This repository provides neural network-based controllers that enhance the performance of stable nonlinear systems while maintaining stability guarantees.

## 📋 Overview

The performance boosting controller is designed to:
- Improve tracking performance of pre-stabilized nonlinear systems
- Maintain stability guarantees through contractive neural network architectures such as Recurrent Equilibrium Networks (RENs) and State Space Models (SSMs)
- Handle both linear time-invariant (LTI) and nonlinear robotic systems
- Support various neural network architectures including RENs and SSMs

## ✨ Features

- **Neural Network Controllers**: Support for RENs and SSMs with different scaffolding nonlinearities
- **Multiple Plant Types**: 
  - Linear Time-Invariant (LTI) systems
  - Multi-agent robotic systems with collision avoidance
- **Stability Guarantees**: Contractive REN implementation ensures stability
- **Flexible Architecture**: Multiple scaffolding options for SSMs (MLPs, coupling layers, Hamiltonian NNs)
- **GPU Support**: CUDA and Apple Silicon (MPS) acceleration

## 🚀 Installation

### Requirements

- Python ≥ 3.10.12
- PyTorch ≥ 2.2
- NumPy ≥ 1.24.4
- SciPy ≥ 1.13.1
- Matplotlib == 3.8.4

### Setup

1. Clone the repository:
```bash
git clone https://github.com/DecodEPFL/perf-boost-base.git
cd perf-boost-base
```

2. Install the package:
```bash
pip install -e .
```

## ⚡ Quick Start

### Minimal Example

Run the minimal example with robotic systems:

```bash
cd experiments/minimal_example
python run.py
```

This will:
- Train a performance boosting controller on a 2-agent robotic system
- Use SSM with tanh nonlinearity by default
- Save results and plots in `saved_results/`

### LTI System Example

For linear time-invariant systems:

```bash
cd experiments/LTI
python run.py
```

## 🏗️ Architecture

### Core Components

#### Performance Boosting Controller (`controllers/PB_controller.py`)
The main controller class that implements:
- State feedback control with memory
- Integration with REN or SSM neural networks
- Noise reconstruction and compensation
- Training loop with validation and early stopping

#### Neural Network Options

1. **Contractive REN** (`controllers/contractive_ren.py`)
   - Provides stability guarantees through contraction constraints
   - Acyclic architecture with internal states

2. **State Space Models** (`controllers/ssm.py`)
   - Deep SSM implementation with Linear Recurrent Units (LRU)
   - Multiple scaffolding options:
     - MLPs (`controllers/non_linearities.py`)
     - Hamiltonian Neural Networks
     - Coupling layers

#### Plant Models

1. **LTI Systems** (`plants/LTI/`)
   - Standard linear time-invariant plant model
   - Support for arbitrary A, B, C matrices

2. **Robotic Systems** (`plants/robots/`)
   - Multi-agent robotic systems with collision avoidance
   - Nonlinear dynamics with friction terms
   - Customizable number of agents and parameters

#### Loss Functions

- **LQ Loss** (`loss_functions/lq_loss.py`): Standard Linear Quadratic cost
- **Robots Loss** (`loss_functions/robots_loss.py`): Multi-agent cost with collision avoidance and obstacle penalties

## 💻 Usage

### Basic Controller Setup

```python
from controllers.PB_controller import PerfBoostController
from plants.robots import RobotsSystem

# Define system
sys = RobotsSystem(xbar=nominal_point, linear_plant=False)

# Create controller
controller = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init,
    output_init=sys.u_init,
    nn_type="SSM",  # or "REN"
    scaffolding_nonlin="tanh",  # for SSMs
    dim_internal=8,
    dim_nl=8
)

# Train controller
controller.fit(
    sys=sys,
    train_dataloader=train_loader,
    valid_data=validation_data,
    lr=1e-3,
    loss_fn=loss_function,
    epochs=500
)
```

### Configuration Options

#### Neural Network Types
- `"REN"`: Contractive Recurrent Equilibrium Network
- `"SSM"`: State Space Model

#### SSM Scaffolding Nonlinearities
- `"tanh"`: Standard tanh activation
- `"hamiltonian"`: Hamiltonian Neural Network
- `"coupling_layers"`: Coupling layer architecture

#### Training Parameters
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `batch_size`: Batch size for training
- `early_stopping`: Enable early stopping based on validation loss
- `return_best`: Return best model based on validation performance

## 🔬 Examples

### Multi-Agent Robotics

The robotic system example demonstrates:
- 2-agent coordination
- Collision avoidance
- Obstacle navigation
- Performance improvement over baseline controller

Key parameters:
- `n_agents`: Number of robotic agents
- `min_dist`: Minimum inter-agent distance
- `alpha_col`: Collision avoidance weight
- `alpha_obst`: Obstacle avoidance weight

### LTI Systems

For control of linear systems with:
- Custom A, B, C matrices
- Process noise handling
- LQ cost optimization

## 📊 Results and Visualization

The package includes plotting utilities for:
- Trajectory visualization
- Training loss curves
- Before/after training comparisons
- Multi-agent coordination plots

Results are automatically saved in timestamped directories under `experiments/*/saved_results/`.

## 🖥️ Device Support

Automatic device selection:
- CUDA GPU (if available)
- Apple Silicon MPS (if available)
- CPU (fallback)

## 📜 License

This project is licensed under the CC-BY-4.0 License.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{performance_boosting_controllers,
  title={Learning to Boost the Performance of Stable Nonlinear Systems},
  author={[Author names]},
  journal={[Journal name]},
  year={[Year]}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions and support, please contact: [mahrokh.ghoddousiboroujeni@epfl.ch]