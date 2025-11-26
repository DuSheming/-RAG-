# 安装指南

## 📦 快速安装

### 在本地环境

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd patent-rag

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements_full.txt
```

### 在Google Colab

```python
# 一行命令安装所有依赖
!pip install faiss-cpu sentence-transformers rank-bm25 openai psutil -q

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

## 🔧 依赖说明

### 核心依赖（必需）

| 包名 | 版本 | 用途 |
|------|------|------|
| faiss-cpu | 1.7.4 | 向量检索引擎 |
| sentence-transformers | ≥2.2.2 | 文本向量化 |
| torch | ≥2.0.0 | 深度学习框架 |
| numpy | ≥1.24.0 | 数组计算 |
| tqdm | ≥4.65.0 | 进度条 |

### 高级功能（可选）

| 包名 | 版本 | 用途 |
|------|------|------|
| rank-bm25 | ≥0.2.2 | BM25检索 |
| openai | ≥1.0.0 | DeepSeek API |
| psutil | ≥5.9.0 | 内存监控 |

## 💾 磁盘空间需求

- **模型缓存**：~500MB
  - `all-MiniLM-L6-v2`: ~90MB
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`: ~90MB

- **索引文件**：~250MB
  - 10个小索引，每个约25MB

- **临时文件**：~100MB

**总计**：约1GB

## 🖥️ 硬件要求

### 最低配置

- **CPU**: 2核
- **RAM**: 12GB
- **存储**: 2GB

### 推荐配置

- **CPU**: 4核+
- **RAM**: 16GB+
- **存储**: 5GB+
- **GPU**: 可选（NVIDIA + 4GB VRAM）

## 🐛 常见安装问题

### 问题1: FAISS安装失败

```bash
# 错误信息
# ERROR: Could not find a version that satisfies the requirement faiss-cpu

# 解决方案：使用conda安装
conda install -c conda-forge faiss-cpu

# 或使用预编译轮子
pip install faiss-cpu --no-cache-dir
```

### 问题2: Torch版本不兼容

```bash
# 根据CUDA版本安装对应的PyTorch
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 问题3: 内存不足

```bash
# 安装时内存不足，可以分步安装
pip install faiss-cpu
pip install sentence-transformers
pip install rank-bm25 openai psutil
```

### 问题4: 模型下载慢

```bash
# 使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com
pip install sentence-transformers

# 或手动下载模型到缓存目录
# ~/.cache/huggingface/hub/
```

## 🚀 验证安装

运行以下代码验证安装：

```python
# test_installation.py

import faiss
import sentence_transformers
import numpy as np
from rank_bm25 import BM25Okapi

print("✓ FAISS版本:", faiss.__version__)
print("✓ Sentence-Transformers版本:", sentence_transformers.__version__)
print("✓ NumPy版本:", np.__version__)
print("✓ BM25已安装")

# 测试FAISS
dimension = 384
index = faiss.IndexFlatIP(dimension)
vectors = np.random.random((100, dimension)).astype('float32')
faiss.normalize_L2(vectors)
index.add(vectors)
print(f"✓ FAISS测试通过: {index.ntotal} 个向量")

# 测试向量化
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("test")
print(f"✓ 向量化测试通过: 维度 {len(embedding)}")

print("\n🎉 所有依赖安装成功！")
```

运行测试：

```bash
python test_installation.py
```

## 📱 Docker部署（可选）

```dockerfile
# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements_full.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements_full.txt

# 复制项目文件
COPY . .

# 暴露端口（如果有Web服务）
EXPOSE 8080

CMD ["python", "your_main_script.py"]
```

构建和运行：

```bash
docker build -t patent-rag .
docker run -it --rm -v $(pwd)/data:/app/data patent-rag
```

## 🔄 更新依赖

```bash
# 更新所有包到最新版本
pip install --upgrade -r requirements_full.txt

# 更新特定包
pip install --upgrade sentence-transformers

# 查看已安装版本
pip list | grep -E "faiss|sentence|torch"
```

## 💡 性能优化建议

### CPU优化

```bash
# 使用更多线程
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### GPU优化

```bash
# 安装GPU版FAISS
pip uninstall faiss-cpu
pip install faiss-gpu

# 验证CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 内存优化

```python
# 在代码中限制batch size
batch_size = 8  # 降低到8（从默认32）
```

---

✅ **安装完成后，查看 [README.md](README.md) 开始使用！**
