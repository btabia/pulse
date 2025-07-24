# Pushing Unstructured Loose Soft Elements

This tutorial guides you through setting up the environment for the PULSE project, including Isaac Sim, Isaac Lab, and key dependencies.

---

## 1. Install Isaac Sim

### 1.1 Create a Conda Environment

```bash
conda create -n pulse python=3.10
conda activate pulse
```

### 1.2 Install CUDA-enabled PyTorch

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### 1.3 Upgrade pip

```bash
pip install --upgrade pip
```

### 1.4 Install Isaac Sim Packages

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

### 1.5 Verify Isaac Sim Installation

```bash
isaacsim --help
```

You should see a list of available arguments.

#### To run Isaac Sim remotely in streaming mode:

```bash
isaacsim isaacsim.exp.full.streaming --no-window
```

---

## 2. Install Isaac Lab

### 2.1 Pull the Bespoke Isaac Lab Submodule

```bash
git submodule update --recursive --force
```

### 2.2 Install System Dependencies

```bash
sudo apt install cmake build-essential
```

### 2.3 Install Isaac Lab Framework

```bash
./isaaclab.sh --install skrl
```

---

## 3. Install Additional Python Dependencies

### 3.1 Install wandb

```bash
pip install wandb
```

**Note:**  
To avoid issues with `wandb` and `sentry_sdk`, remove the sentry component bundled with Isaac Sim:

```bash
cd /anaconda3/envs/pulse/lib/python3.10/site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.0+d02c707b.lx64.cp310/pip_prebundle
sudo rm -rf sentry_sdk sentry_sdk-1.43.0.dist-info/
```

### 3.2 Install open3d

```bash
pip install open3d
```

### 3.3 Install PyTorch Geometric

```bash
pip install torch_geometric
```

---

You are now ready to use the PULSE environment!








