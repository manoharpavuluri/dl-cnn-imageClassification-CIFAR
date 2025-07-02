# Setup Guide for CIFAR-10 CNN Image Classification

This guide provides detailed instructions for setting up the CIFAR-10 CNN Image Classification project on different operating systems.

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for faster training

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or higher
- **Storage**: 5GB free space
- **GPU**: NVIDIA GPU with CUDA support

## 🐍 Python Installation

### Windows

1. **Download Python**
   - Visit [python.org](https://www.python.org/downloads/)
   - Download Python 3.9 or 3.10
   - Run the installer with "Add Python to PATH" checked

2. **Verify Installation**
   ```cmd
   python --version
   pip --version
   ```

### macOS

1. **Using Homebrew (Recommended)**
   ```bash
   brew install python@3.9
   ```

2. **Using Official Installer**
   - Download from [python.org](https://www.python.org/downloads/)
   - Run the installer

3. **Verify Installation**
   ```bash
   python3 --version
   pip3 --version
   ```

### Ubuntu/Linux

1. **Install Python**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Verify Installation**
   ```bash
   python3 --version
   pip3 --version
   ```

## 🚀 Project Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/manoharpavuluri/dl-cnn-imageClassification-CIFAR.git
cd dl-cnn-imageClassification-CIFAR
```

### Step 2: Create Virtual Environment

#### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import keras; print('Keras version:', keras.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
```

## 🔧 GPU Setup (Optional but Recommended)

### NVIDIA GPU Setup

1. **Install NVIDIA Drivers**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
   - Install the appropriate driver for your GPU

2. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation guide for your OS

3. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA installation directory

4. **Install GPU-enabled TensorFlow**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow[gpu]
   ```

5. **Verify GPU Setup**
   ```python
   import tensorflow as tf
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   ```

## 📓 Jupyter Notebook Setup

### Install Jupyter
```bash
pip install jupyter notebook
```

### Launch Jupyter
```bash
jupyter notebook
```

### Alternative: JupyterLab
```bash
pip install jupyterlab
jupyter lab
```

## 🐛 Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues

**Problem**: `tensorflow` installation fails
**Solution**:
```bash
pip install --upgrade pip
pip install tensorflow==2.10.0
```

#### 2. Memory Issues

**Problem**: Out of memory errors during training
**Solutions**:
- Reduce batch size in the notebook
- Use data generators for memory efficiency
- Close other applications to free RAM

#### 3. GPU Not Detected

**Problem**: TensorFlow doesn't see GPU
**Solutions**:
```bash
# Check GPU availability
nvidia-smi

# Reinstall TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[gpu]
```

#### 4. Import Errors

**Problem**: Module not found errors
**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

#### 5. Jupyter Kernel Issues

**Problem**: Jupyter can't find Python kernel
**Solution**:
```bash
# Install ipykernel
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=venv --display-name="Python (venv)"
```

### Performance Optimization

#### 1. Enable Mixed Precision Training
```python
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

#### 2. Optimize Memory Usage
```python
# Reduce memory fragmentation
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### 3. Use Data Generators
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

## 🔍 Verification Steps

After setup, run these commands to verify everything is working:

```python
# Test imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

print("All imports successful!")

# Test GPU
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)

# Test basic TensorFlow operations
x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[5, 6], [7, 8]])
z = tf.matmul(x, y)
print("TensorFlow test:", z.numpy())
```

## 📚 Additional Resources

### Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### Tutorials
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)

### Community
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow)

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. **Check the Issues**: Look at existing GitHub issues
2. **Search Documentation**: Check TensorFlow and Keras docs
3. **Community Forums**: Post on TensorFlow forum or Stack Overflow
4. **Create Issue**: Open a new GitHub issue with detailed information

### When Creating an Issue, Include:
- Operating system and version
- Python version
- TensorFlow version
- Complete error message
- Steps to reproduce the issue
- System specifications (RAM, GPU, etc.)

---

**Happy Coding! 🚀** 