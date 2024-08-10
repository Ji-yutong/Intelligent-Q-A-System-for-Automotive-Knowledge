# 汽车知识问答系统

这是一个基于 `chatglm3-6b` 模型的汽车知识问答系统，集成了 RAG模块、多模态处理、Agent、知识图谱等技术，以提高大模型的回答质量和系统的智能性。系统支持文本和图片输入，能够提供详细的汽车相关信息。

## 如何部署项目

1. **配置文件**：请确保在 `config.json` 中配置了正确的模型路径及参数。配置文件的详细说明见下节“各模块细节信息”。

2. **安装环境依赖**：在 Linux 命令行中运行以下命令来安装所需的 Python 包：

```python
pip install -r requirements.txt
```

3. **运行程序**：在 Linux 命令行中运行以下命令启动程序：

```python
python app.py
```

##各模块细节信息

###1.构建本地知识库

**文件**：vectorize_and_index.py

**模块功能**：该模块负责读取 PDF 文件内容，使用指定的嵌入模型将文本进行分割和编码，并将编码结果存储为 FAISS 索引。RAG 模块将利用这些索引与用户输入进行匹配，从而获取相关的补充信息。可以通过调整 `config['chunk_size']` 参数来改变文本块的大小，以保证上下文的完整性。

**embedding模型**：[sentence-transformers](https://huggingface.co/sentence-transformers)/[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**config 参数**：

- `config['embedding_model_path']`：本地向量库编码模型文件路径。
- `config['vector_dbindex_path']`：FAISS 索引文件路径。
- `config['vector_vectors_path']`：FAISS 向量数据文件路径。
- `config['text_mapping_path']`：文本块与索引映射关系文件路径。
- `config['rag_data_path']`：本地知识库文档路径。
- `config['chunk_size']`：本地文档切分块的大小。

### 2.Agent模块

**文件**：agent_module.py

**模块功能**：初始化系统中的各个模块，并处理用户查询。

**维护对话历史**：记录用户输入与系统回复，使用 mT5 模型生成对话摘要。

**处理用户查询**：利用 Elasticsearch 搜索用户问题。

**管理用户输入与各模块的交互**：

- 使用多模态模块处理用户上传的图片，并提取图像描述。
- 使用 RAG 模块从本地向量库中检索与用户输入问题相关的文档片段。
- 使用知识图谱模块对 RAG 匹配的文档进行增强。
- 将用户输入问题、历史对话摘要、RAG 匹配的文档、知识图谱增强的内容整合到 prompt 中，并送入大模型生成最终回复。

**LLM**：[THUDM](https://huggingface.co/THUDM)/[chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)

**config参数**：

- `config['elasticsearch_host']`：Elasticsearch 服务器地址。
- `config['llm_path']`：chatglm3-6b 模型和分词器的路径。
- `config['vector_dbindex_path']`：FAISS 索引文件路径。
- `config['vector_vectors_path']`：FAISS 向量数据文件路径。
- `config['summary_model_path']`：mt5 摘要模型和分词器的路径。

###3.多模态模块

**文件**：rag_module.py.py

**模块功能**：加载 BLIP 模型，将图像文件转换为文本描述，支持系统对图像内容的理解和处理。

**BLIP模型**：[Salesforce](https://huggingface.co/Salesforce)/[blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large)

**config参数**：

- `config['blip_model_path']`：BLIP模型的本地路径。

###4.rag模块

**文件**：multimodal_module.py

**模块功能**：加载 FAISS 索引和嵌入模型（应与构建本地知识库时使用的模型一致），对输入的查询文本进行向量编码，并从本地向量库中检索最相关的文本片段。通过 `config['rag_retrieval_top_k']` 参数可以调整检索的文档数量，以提供更多上下文信息来增强大模型的回复质量。

**embedding模型**：[sentence-transformers](https://huggingface.co/sentence-transformers)/[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**config参数**：

- `config['embedding_model_path']`：编码模型的本地路径。
- `config['vector_dbindex_path']`：FAISS 索引文件路径。
- `config['rag_retrieval_top_k']`：检索时返回的最匹配的向量块个数。
- `config['text_mapping_path']`：文本块与索引的映射关系文件路径。

### 5.知识图谱模块

**文件**：knowledge_graph.py

**模块功能**：通过提取上下文中的关键词并在维基百科上进行搜索来增强上下文信息。流程如下：

1. **分词**：使用 jieba 对上下文进行中文分词，将文本拆分为词或短语。
2. **TF-IDF 筛选**：使用 TF-IDF 技术筛选上下文中的关键词，并提取 TF-IDF 分数最高的前几个词。
3. **BERT 嵌入**：加载 BERT 模型生成这些关键词的嵌入向量，并计算词嵌入与句子嵌入之间的余弦相似度，提取最相关的关键词。
4. **维基百科搜索**：使用维基百科 API 对提取的关键词进行搜索，并将搜索结果与上下文结合，增强信息。

**bert模型**：[huawei-noah](https://huggingface.co/huawei-noah)/[TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)

**config参数**：

- `config['tiny_bert_model_path']`：BERT 模型和分词器的本地路径。
- `config['knowledge_graph_search_number']`：在维基百科上进行搜索时返回的结果数量。





































































































