# 文献专利 RAG 检索系统

基于 BGE-M3 + FAISS + BM25 的多模态文献专利检索系统，支持混合检索、重排序和 LLM 问答。系统经历了 4 个版本的迭代，从基础 RAG 逐步演进为支持结构化文档解析、多模态生成和自动化评估的完整系统。

## 版本演进

| 版本 | 核心特性 | 关键技术 |
|------|----------|----------|
| **v1** | 基础 RAG | FAISS 向量检索 + BM25 关键词检索 + 多索引架构 |
| **v2** | 多模态检索 | BGE-M3 三路检索（Dense + Sparse + BM25）+ RRF 融合 + Cross-Encoder 重排序 + 图片 caption 处理 |
| **v3** | 稀疏检索加速 + 评估体系 | Sparse 存储升级 CSR 矩阵（快 100x）+ VLM 图片分级处理 + RAGAS 四维评估框架 |
| **v4** | 结构化解析 + 多模态生成 | content_list.json 结构化解析 + 文档层级树 + 表格三重表示 + Parent-Child Chunking + 多模态 LLM 生成 |

## 系统架构（v4）

```
用户查询
    │
    ▼
┌─────────────────────────────────────────────┐
│          LiteratureRAG_v4 检索引擎           │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Dense   │  │ Sparse   │  │  BM25     │  │
│  │ FAISS   │  │ CSR矩阵  │  │  jieba    │  │
│  │ HNSW    │  │ BGE-M3   │  │  分词     │  │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  │
│       └────────────┼──────────────┘         │
│                    ▼                        │
│              RRF 融合排序                    │
│                    ▼                        │
│           Cross-Encoder 重排序              │
│                    ▼                        │
│         Parent-Child 上下文扩展              │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│            多模态生成 (v4)                    │
│  文本 + 原始表格HTML + 原始图片 → LLM/VLM    │
└─────────────────────────────────────────────┘
```

## 各版本详细说明

### v1 — 基础多索引 RAG

- 将大规模文档分割成多个小索引，避免内存溢出（904 篇文献 → 10 个小索引）
- FAISS 向量检索（语义相似）+ BM25 检索（关键词匹配）
- Cross-Encoder 重排序
- DeepSeek LLM 集成问答
- 支持断点续传和 Google Drive 持久化

### v2 — 多模态混合检索

- 升级为 BGE-M3 编码模型（支持 Dense + Sparse + ColBERT 三种表示）
- 三路检索 + RRF（Reciprocal Rank Fusion）融合
- 图片 caption 文本化处理
- MinerU `full.md` 文档解析 + Markdown 层级切块

### v3 — 稀疏检索加速 + RAGAS 评估

- **Sparse 存储升级**：`List[dict]` Python 遍历 → `scipy.sparse.csr_matrix` 矩阵乘法，检索速度提升约 100 倍
- **VLM 图片分级处理**：caption 足够时走文本，机理图/结构图调 VLM API（GPT-4o / Claude）生成详细描述
- **RAGAS 评估框架**：
  - 自动从文献库生成测试问答对（无需手动标注）
  - 四维指标：context_recall、context_precision、faithfulness、answer_relevancy
  - 快速幻觉检测（无需 ground truth）

### v4 — 结构化解析 + 多模态生成

- **结构化文档解析**：基于 MinerU `content_list.json`，按 block 类型（text/table/image/equation）差异化处理
- **层级文档树**：借鉴 PageIndex 思想，从 MinerU layout 构建 section 树，支持上下文扩展
- **表格三重表示**：结构化行文本（BM25）+ LLM 摘要（Dense）+ 原始 HTML（生成时引用）
- **Parent-Child Chunking**：小 chunk 精确检索 → 回溯 parent 大 chunk 送 LLM，兼顾精度和完整性
- **丰富元数据**：DOI、作者、年份、文档类型、页码，支持 metadata filtering
- **数据清洗**：自动识别并排除非学术文档（发票、收据等）
- **多模态生成**：文本 + 原始表格 + 原始图片一起送多模态 LLM

## 快速开始

### 环境要求

- Python 3.8+
- 16GB+ RAM（推荐，v1 可在 12GB 下运行）
- Google Colab（推荐，免费 GPU）
- DeepSeek API 密钥
- （可选）OpenAI API 密钥（GPT-4o VLM 图片描述）

### 安装依赖

```bash
# v1/v2 基础依赖
pip install faiss-cpu sentence-transformers rank-bm25 openai tqdm

# v3/v4 额外依赖
pip install FlagEmbedding scipy langchain langchain-text-splitters langchain-openai
pip install ragas datasets pillow jieba
pip install lxml beautifulsoup4  # v4 表格处理
```

### 在 Google Colab 中使用

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 建索引（以 v4 为例）
all_chunks, doc_trees = load_all_documents_v4(MINERU_DIR, use_vlm=True, use_table_summary=True)
build_index_v4(all_chunks, doc_trees, INDEX_DIR)

# 3. 加载并查询
rag = LiteratureRAG_v4(index_dir=INDEX_DIR)
result = rag.ask("Which photoinitiators work better under LED light sources?")
print(result['answer'])
```

## 核心技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| Embedding | BGE-M3 (BAAI) | Dense + Sparse 双编码 |
| 向量索引 | FAISS HNSW | 高效近似最近邻搜索 |
| 稀疏索引 | scipy CSR Matrix | 高速稀疏向量点积 |
| 关键词检索 | BM25Okapi + jieba | 中英文关键词匹配 |
| 重排序 | BGE-Reranker-v2-M3 | Cross-Encoder 精排 |
| 文档解析 | MinerU | PDF → 结构化 Markdown/JSON |
| LLM | DeepSeek Chat | 问答生成、表格摘要、评估 |
| VLM | GPT-4o / Claude | 化学图片描述、多模态生成 |
| 评估 | RAGAS | 检索 + 生成质量四维评估 |

## 项目文件

```
├── 文献专利RAG系统.ipynb    # 主文件，包含 v1-v4 所有版本代码
├── RAG.py                  # v1 独立 Python 脚本
├── README.md               # 项目说明
├── INSTALL.md              # 安装指南
└── requirements_full.txt   # 依赖列表
```

## 参考资料

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — 多语言多粒度嵌入模型
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI 向量检索库
- [MinerU](https://github.com/opendatalab/MinerU) — PDF 结构化解析工具
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG 评估框架
- [DeepSeek](https://www.deepseek.com/) — LLM API 服务

## 作者

开发者：Sheming Du

联系方式：862349743@qq.com

## 许可证

MIT License
