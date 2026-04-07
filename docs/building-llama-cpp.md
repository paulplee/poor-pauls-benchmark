# Building and Upgrading llama.cpp

PPB uses [llama.cpp](https://github.com/ggerganov/llama.cpp) as its inference backend. This guide covers building from source, upgrading, and troubleshooting common issues.

## Prerequisites

- **Git**
- **CMake** ≥ 3.14
- **C/C++ compiler** — GCC ≥ 11, Clang ≥ 14, or MSVC 2019+
- **GPU SDK** (for GPU acceleration):
  - NVIDIA: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) ≥ 11.7
  - AMD: [ROCm](https://rocm.docs.amd.com/) ≥ 5.5
  - Apple: Xcode Command Line Tools (Metal is enabled by default on macOS)

## Quick Start

### Clone

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### Build with NVIDIA CUDA

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Build with AMD ROCm (HIP)

```bash
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Build with Apple Metal (macOS)

Metal is enabled by default on macOS — no extra flags needed:

```bash
cmake -B build \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### Build with Vulkan (cross-platform)

```bash
cmake -B build \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### CPU-only (no GPU)

```bash
cmake -B build \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

## CUDA Architecture Targeting

When building with CUDA, you can target specific GPU architectures for optimal performance. Use `CMAKE_CUDA_ARCHITECTURES` to specify the compute capability:

| Compute Capability | GPU Family                                |
| ------------------- | ----------------------------------------- |
| `60`                | Pascal (GTX 1080, Tesla P100)             |
| `70`                | Volta (Tesla V100)                        |
| `75`                | Turing (RTX 2080, T4)                     |
| `80`                | Ampere (RTX 3090, A100)                   |
| `86`                | Ampere (RTX 3060, RTX 3070, A40)          |
| `89`                | Ada Lovelace (RTX 4090, RTX 4080, L40)    |
| `90`                | Hopper (H100, H200)                       |
| `100`               | Blackwell (B100, B200)                    |
| `120`               | Blackwell (RTX 5090, RTX 5080, RTX 5070)  |

Example — targeting an RTX 5090:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build --config Release -j$(nproc)
```

> **Tip:** Omitting `CMAKE_CUDA_ARCHITECTURES` lets CMake auto-detect the installed GPU. Setting it explicitly avoids building kernels for GPU architectures you'll never use, which speeds up compilation.

## Making llama.cpp Available to PPB

After building, you need the `llama-bench` and `llama-server` binaries on your PATH. Three options:

### Option 1: Add the build directory to PATH

```bash
echo 'export PATH="$HOME/source/llama.cpp/build/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Symlink into a PATH directory

```bash
sudo ln -sf ~/source/llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench
sudo ln -sf ~/source/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
```

### Option 3: Use environment variables

```bash
export PPB_LLAMA_BENCH=~/source/llama.cpp/build/bin/llama-bench
export PPB_LLAMA_SERVER=~/source/llama.cpp/build/bin/llama-server
```

Or set them in a TOML suite's `[sweep.runner_params]`:

```toml
[sweep.runner_params]
llama_bench_cmd = "/home/user/source/llama.cpp/build/bin/llama-bench"
# or for llama-server runner:
llama_server_cmd = "/home/user/source/llama.cpp/build/bin/llama-server"
```

### Verify

```bash
llama-server --version
# Expected output:
# version: 8696 (69c28f154)
# built with GNU 13.3.0 for Linux x86_64
```

## Upgrading llama.cpp

New model architectures are added to llama.cpp regularly. If PPB reports an error like `unknown model architecture: 'gemma4'`, you need to upgrade.

### Step-by-step upgrade

```bash
cd ~/source/llama.cpp

# Save any local modifications
git stash

# Pull the latest code
git pull origin master

# Reconfigure (same flags as your original build)
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120    # set to your GPU's compute capability

# Rebuild
cmake --build build --config Release -j$(nproc)

# Verify
llama-server --version
```

### Checking your current build

```bash
# Version and build info
llama-server --version

# What's currently installed
git -C ~/source/llama.cpp log --oneline -1

# Available tags (build numbers)
git -C ~/source/llama.cpp tag --sort=-v:refname | grep "^b[0-9]" | head -5
```

### Checking if a model architecture is supported

```bash
# Try loading a model — if the architecture is unsupported, you'll see:
#   "error loading model architecture: unknown model architecture: 'xxx'"
llama-server -m /path/to/model.gguf -c 512 --host 127.0.0.1 --port 19999 2>&1 | head -20
# Ctrl+C to stop

# Or search the commit history for architecture support
cd ~/source/llama.cpp
git log --oneline --all | grep -i "gemma4"   # example: check for gemma4 support
```

## Minimum Versions for Model Architectures

Some model architectures require a minimum llama.cpp build. If you see `unknown model architecture` errors, upgrade to at least the version listed below:

| Architecture | Models                                   | Minimum Build |
| ------------ | ---------------------------------------- | ------------- |
| `gemma4`     | Gemma 4 E2B, Gemma 4 26B-A4B            | ≥ b8688       |
| `gemma3`     | Gemma 3 (1B–27B)                         | ≥ b5200       |
| `qwen3`      | Qwen 3 / Qwen 3.5                       | ≥ b5500       |
| `deepseek2`  | DeepSeek-R1, DeepSeek-V3                 | ≥ b4700       |
| `llama`      | LLaMA 1/2/3, Mistral, most models       | Any           |

> **Note:** These are approximate minimum versions. When in doubt, use the latest build.

## Useful CMake Options

| Flag                          | Default | Description                                                       |
| ----------------------------- | ------- | ----------------------------------------------------------------- |
| `GGML_CUDA`                   | OFF     | Enable NVIDIA CUDA backend                                        |
| `GGML_HIP`                    | OFF     | Enable AMD ROCm/HIP backend                                      |
| `GGML_METAL`                  | auto    | Enable Apple Metal backend (ON by default on macOS)               |
| `GGML_VULKAN`                 | OFF     | Enable Vulkan backend (cross-platform)                            |
| `GGML_NATIVE`                 | ON      | Optimize for the current CPU (sets `-march=native`)               |
| `GGML_CUDA_FA`                | ON      | Compile FlashAttention CUDA kernels                               |
| `GGML_CUDA_FA_ALL_QUANTS`     | OFF     | Compile FA for all quantization types (slower build, wider support) |
| `CMAKE_BUILD_TYPE`            | —       | `Release` (optimized), `Debug`, `RelWithDebInfo`                  |
| `CMAKE_CUDA_ARCHITECTURES`    | auto    | Target GPU compute capability (e.g. `89` for RTX 4090)           |
| `BUILD_SHARED_LIBS`           | OFF     | Build shared libraries instead of static                          |
| `LLAMA_BUILD_SERVER`          | ON      | Build `llama-server` (required for PPB's server runners)          |
| `LLAMA_OPENSSL`               | ON      | Enable HTTPS support in llama-server                              |

## Troubleshooting

### "unknown model architecture"

```
llama_model_load: error loading model: unknown model architecture: 'gemma4'
```

Your llama.cpp build is too old for this model. Upgrade to the latest build (see [Upgrading](#upgrading-llamacpp) above).

PPB detects this automatically during vram-cliff probing and reports the unsupported architecture name along with a suggestion to upgrade.

### Build fails with CUDA errors

Ensure your CUDA toolkit version matches what llama.cpp expects. Check the [llama.cpp build docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) for the latest requirements.

```bash
nvcc --version       # Check installed CUDA version
nvidia-smi           # Check driver version
```

### Server crashes immediately (not OOM)

If every model probe fails in under 3 seconds, the issue is likely not OOM but a model load failure. PPB's vram-cliff now detects this: it probes at the minimum context size first and reports the actual server stderr if the model can't load.

Common causes:
- Unsupported model architecture (upgrade llama.cpp)
- Corrupt GGUF file (re-download the model)
- Missing CUDA libraries (check `LD_LIBRARY_PATH`)

### Slow compilation

- Install `ccache` to cache compiled objects across rebuilds:
  ```bash
  sudo apt install ccache    # Ubuntu/Debian
  brew install ccache         # macOS
  ```
- Target only your GPU architecture with `CMAKE_CUDA_ARCHITECTURES` instead of building for all architectures.
- Use `-j$(nproc)` to parallelize the build across all CPU cores.
