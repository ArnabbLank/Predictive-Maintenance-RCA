# Virtual Environment Setup Guide

## Prerequisites
- **Python 3.10+** (recommended: 3.10.x or 3.11.x)
- **pip** (comes with Python)
- **git** (for cloning the repo)

---

## Option 1: `venv` (built-in, recommended)

### macOS / Linux
```bash
# 1. Navigate to the project root (after cloning)
cd rul-copilot

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=rul-copilot --display-name "RUL Copilot"

# 7. When done, deactivate
deactivate
```

### Windows (PowerShell)
```powershell
# 1. Navigate to the project root (after cloning)
cd rul-copilot

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=rul-copilot --display-name "RUL Copilot"

# 7. When done, deactivate
deactivate
```

### Windows (CMD)
```cmd
.venv\Scripts\activate.bat
```

---

## Option 2: `conda`

```bash
# 1. Create a conda environment
conda create -n rul-copilot python=3.10 -y

# 2. Activate
conda activate rul-copilot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register Jupyter kernel
python -m ipykernel install --user --name=rul-copilot --display-name "RUL Copilot"

# 5. When done
conda deactivate
```

---

## Verifying the Setup

```bash
# Check Python version
python --version          # Should be 3.10+

# Check key packages
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pandas; print(f'pandas {pandas.__version__}')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"

# Launch Jupyter to test notebooks
jupyter notebook
```

---

## PyTorch with GPU (optional)

If you have an NVIDIA GPU and want CUDA support, replace the PyTorch lines in `requirements.txt` or install separately:

```bash
# CUDA 12.1 (check your driver version first with: nvidia-smi)
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

For Apple Silicon (M1/M2/M3), the default `pip install torch` already includes MPS support — no extra steps needed.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip install` fails on `torch` | Install PyTorch separately first, then the rest |
| Jupyter can't find the kernel | Re-run `python -m ipykernel install ...` |
| `ModuleNotFoundError` in notebook | Make sure the notebook is using the `RUL Copilot` kernel |
| `shap` install fails | Try `pip install shap --no-build-isolation` |
| M1/M2 Mac issues | Use `conda` — it handles ARM packages better |

---

## VS Code Integration

1. Open the project folder in VS Code
2. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose `.venv/bin/python` (or the conda env)
4. For notebooks, select "RUL Copilot" as the kernel in the top-right
