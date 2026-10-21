# Setting up the cuscope dev environment (WSL2)

cuscope is developed and tested in WSL2 Ubuntu 22.04 on an RTX 3060 Laptop GPU.
Everything below runs **inside WSL**.

## What you already have

The Windows NVIDIA driver provides the GPU to WSL. `libcuda.so.1` lives in
`/usr/lib/wsl/lib`, and `nvidia-smi` works inside WSL.

> **Never install a Linux NVIDIA driver inside WSL.** Do not install the `cuda`,
> `cuda-drivers` or `nvidia-driver-*` packages; they overwrite the WSL driver
> stub and break GPU access. Only the toolkit package below is safe.

Check the GPU first:

```bash
nvidia-smi
```

It should list `NVIDIA GeForce RTX 3060 Laptop GPU` with 6144 MiB.

## 1. CUDA toolkit 13.0 (pinned)

cuscope supports exactly one toolkit version, **CUDA 13.0**. It matches the
CUDA runtime bundled with PyTorch 2.14+cu130, so every program requests the same
driver function versions. The download is several GB.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

## 2. Build tools and test runner

Ubuntu 22.04 ships CMake 3.22, and cuscope needs 3.28 or newer for CUDA 13:

```bash
python3 -m pip install --user "cmake>=3.28,<4" ninja pytest
```

## 3. PATH

```bash
echo 'export PATH=/usr/local/cuda-13.0/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## 4. Verify

Open a new WSL shell and run:

```bash
nvcc --version                       # ... release 13.0 ...
cmake --version                      # 3.28 or newer
ninja --version
python3 -m pytest --version
ls /usr/local/cuda-13.0/include/cuda.h
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.14.0+cu130 True
```

## 5. Build cuscope

```bash
cd ~/cuscope
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```
