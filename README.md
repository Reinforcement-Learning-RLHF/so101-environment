# SO-101 Pouring Environment

This repository contains an environment for the **SO-101 robot arm** performing the *cup pouring task*.  
It supports **collecting demos**, **pushing datasets and policies to the LeRobot Hub**, and **running pretrained policies in MuJoCo simulation**.

---

## 🛠 Installation & Setup

### 1. Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [MuJoCo](https://mujoco.org/) installed and properly configured
- Optional: GPU for faster inference with PyTorch

### 2. Create Python Environment
```bash
# Create Python 3.10 environment
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

Run the scripted controller to generate demo trajectories:

```bash
python scripts/collection.py
```

* Generates trajectories of the SO-101 pouring task
* Stores images, joint states, and actions for each frame
* Saves locally under `data/lerobot/so101_pouring`

### 2. Push Dataset to LeRobot Hub

After collecting demos, upload the dataset so your team or the public can access it:

```bash
python scripts/push_data.py
```

* Dataset is now available publicly at `https://huggingface.co/username/so101_pouring`
* Others can load it directly using `LeRobotDataset("username/so101_pouring")`

### 3. Push Policy to LeRobot Hub

Pretrained policies can also be uploaded and shared:

```bash
python scripts/push_policy.py
```

* Policy is now available publicly at `https://huggingface.co/username/so101_act_policy`
* Anyone can use it directly with `ACTPolicy.from_pretrained("username/so101_act_policy")`

### 4. Run Inference in MuJoCo (Plug-and-Play from Hub)

You can run a pretrained policy **without any local files**, fully from Hugging Face:

```bash
python scripts/inference.py
```

Key features:

* Automatically loads both **policy** and **dataset statistics** from the Hub
* Renders the SO-101 pouring task in MuJoCo simulation
* Handles joint states and front/wrist camera observations
* Resets environment and policy on episode termination or truncation

---

## 📁 Project Structure

* **envs/** — Robot environment definitions for MuJoCo
* **models/** — SO-101 XML and 3D assets
* **scripts/** — Utility scripts for demo collection, inference, visualization
* **data/** — Local storage for datasets
* **setup.py** — Package installation configuration

---

## 🧪 Features

* **Scripted demo collection** for SO-101 pouring
* **LeRobot Hub integration** for dataset & policy storage and sharing
* **MuJoCo inference** for testing pretrained policies
* Supports both **human visualization** and **automated evaluation**
* Plug-and-play usage: policies and datasets can be loaded directly from Hugging Face Hub

---

## 👥 Getting Started for Teammates

1. Install dependencies and activate environment.
2. Run `scripts/inference.py` to visualize the SO-101 pouring task in simulation.
   No local files needed — everything is pulled automatically from the Hub.
