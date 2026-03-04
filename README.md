# Preference Learning SO-101

This repository focuses on developing preference learning algorithms for the **SO-101 robot arm**. The project involves training reward models and implementing imitation learning techniques.

## 🛠 Installation & Setup

### 1. Prerequisites
Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed.

### 2. Create Environment
```bash
# Create and activate a new python 3.10 environment
conda create -n so101 python=3.10 -y
conda activate so101

```

### 3. Install Dependencies

```bash
# Install required libraries
pip install -r requirements.txt

# Install the project in editable mode (Fixes ModuleNotFoundError)
pip install -e .

```

---

## 🚀 Usage

### Visualization

To verify the MuJoCo model and environment setup are working correctly, run the visualization script:

```bash
python scripts/visualize.py

```

---

## 📁 Project Structure

* **`envs/`**: Core environment definitions, including `ArmEnv` for MuJoCo interfacing.
* **`models/`**: MuJoCo XML configuration files and 3D mesh assets for the SO-101.
* **`scripts/`**: Utility scripts for testing, visualization, and data collection.
* **`setup.py`**: Configuration for installing the project as a local Python package.

---

## 🧪 Ongoing Research

* **Preference Learning**: Training agents based on human feedback on the SO-101 platform.
* **Policy Architectures**: Implementing imitation learning policies for baseline robot control.