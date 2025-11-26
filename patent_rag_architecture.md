# 专利/文献多模态RAG系统架构

## 数据结构分析

MinerU输出包含：
```
文献目录/
├── full.md                  # 完整markdown文本
├── layout.json              # 页面布局信息
├── content_list.json        # 结构化内容列表（重要！）
└── images/                  # 提取的图像
    ├── 化学结构式.jpg
    ├── 反应scheme.jpg
    └── 表格图像.jpg
```

## 推荐方案：分层多模态索引

### 方案A：基于content_list.json的精细化切分（推荐）

**优势**：最大化保留信息，支持精确检索

```python
# 数据结构
{
    "chunk_id": "uuid",
    "type": "text|image|table",
    "content": "文本内容或图像描述",
    "page_idx": 0,
    "text_level": 1,  # 标题层级
    "image_path": "images/xxx.jpg",  # 如果是图像
    "vector": [...],  # embedding向量
    "metadata": {
        "doc_id": "Liu_et_al-2015",
        "title": "论文标题",
        "authors": ["Liu", "Kim"],
        "year": 2015,
        "journal": "Advanced Synthesis & Catalysis",
        "section": "Introduction",  # 章节
        "keywords": ["copper catalyst", "aryl thiols"]
    }
}
```

### 方案B：混合语义块切分（平衡方案）

适合计算资源有限的情况，合并相关段落。

### 方案C：大块+小块双索引（高级方案）

- 小块：精确检索
- 大块：提供完整上下文

## 核心技术选型

### 1. 文本Embedding
- **科学文献专用**：`allenai/specter2` (推荐)
- **通用强大**：`text-embedding-3-large` (OpenAI)
- **开源替代**：`BAAI/bge-large-zh-v1.5` (中英文)

### 2. 图像处理（化学结构式）
- **多模态Embedding**：`openai/clip-vit-large-patch14`
- **图像描述生成**：GPT-4V / Claude 3 Vision
- **化学专用**：`MolCLIP` (分子结构识别)

### 3. 向量数据库
- **开源**：Qdrant, Milvus
- **云服务**：Pinecone, Weaviate
- **轻量级**：ChromaDB, FAISS

### 4. 检索策略
- 混合检索：向量检索 + BM25关键词检索
- 重排序：使用reranker模型提升准确度
- 多模态融合：文本+图像联合检索

## 信息最大化保留策略

### 文本处理
1. **保留化学公式**：LaTeX格式保持完整
2. **保留表格结构**：转为结构化JSON
3. **章节层级**：使用text_level字段
4. **引用关系**：提取文献引用[1],[2]

### 图像处理
1. **自动分类**：
   - 化学结构式 → 使用SMILES转换
   - 反应Scheme → 提取反应条件
   - 表格图像 → OCR识别
   - 曲线图 → 数据提取

2. **多重索引**：
   - 图像原始embedding
   - GPT-4V生成的详细描述
   - 提取的结构化信息

### 元数据增强
- 自动提取关键词
- 识别化合物名称
- 提取实验条件（温度、催化剂、溶剂）
- 识别合成路线

## 查询增强

### Query Expansion
- 化学名称同义词扩展
- 相关反应类型联想
- 多语言查询支持

### Hypothetical Document Embeddings (HyDE)
使用LLM生成假设答案再检索，提升召回率
