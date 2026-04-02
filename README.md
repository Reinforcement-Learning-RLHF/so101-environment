# SO-101 Pouring Environment

This repository contains an environment for the **SO-101 robot arm** to perform a *cup pouring task*.  
It supports collecting demos using a scripted controller, pushing datasets to the LeRobot Hub, and running inference in MuJoCo simulation.

---

## 🛠 Installation & Setup

### 1. Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [MuJoCo](https://mujoco.org/) installed and properly configured

### 2. Create Environment
```bash
# Create a Python 3.10 environment
conda create -n so101 python=3.10 -y
conda activate so101
````

### 3. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

---

## 🚀 Usage

### 1. Collect Demos

Collect demos of the SO-101 pouring using the scripted controller:

```bash
python scripts/collection.py
```

This will generate a dataset of trajectories you can later use for training or analysis.

### 2. Push Dataset to LeRobot Hub

After collecting demos, push your dataset to the LeRobot Hub

### 3. Run Inference in MuJoCo

Load a trained policy and run it in simulation:

```bash
python scripts/inference.py
```

---

## 📁 Project Structure

* **envs/**: Robot environment definitions for MuJoCo
* **models/**: SO-101 XML and 3D assets
* **scripts/**: Utility scripts for demo collection, visualization, and evaluation
* **data/**: Folder to store local datasets
* **setup.py**: Local package installation configuration

---

## 🧪 Features

* Scripted demo collection for the SO-101 pouring task
* Integration with the LeRobot Hub for dataset storage and sharing
* MuJoCo inference for testing policies in simulation
* Support for both human visualization and automated evaluation
