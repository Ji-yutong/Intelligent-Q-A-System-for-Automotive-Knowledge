import faiss
import json

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# 在新环境中解决冲突问题，去掉下面代码后是否报错？需要一个正确的库版本
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 加载配置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)


class RAGModule:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config['embedding_model_path'])    # 加载tokenizer
        self.model = AutoModel.from_pretrained(config['embedding_model_path'])            # 加载model
        self.index = faiss.read_index(config['vector_dbindex_path'])                      # 加载FAISS索引文件
        self.top_k = config['rag_retrieval_top_k']                                        # 加载top_k

        # 加载文本块与索引的映射
        with open(config['text_mapping_path'], 'r', encoding='utf-8') as f:
            self.text_mapping = json.load(f)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] token 的输出作为句子向量
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def query(self, query_text):
        query_vector = self.encode_text([query_text])  # 将查询文本转换为向量
        distances, indices = self.index.search(query_vector, self.top_k)  # 在FAISS索引中搜索最接近的top_k个向量
        return indices  # 返回最接近的向量的索引

    def get_texts_from_indices(self, indices):
        # 根据索引从加载的文本数据中提取实际文本
        return [self.text_mapping[str(idx)] for idx in indices]


if __name__ == "__main__":
    rag_module = RAGModule()                            # 实例化RAG模块
    test_query = "汽车轮胎漏气时应该如何处理？"              # 定义测试查询文本
    rag_indices = rag_module.query(test_query)              # 执行查询

    # 从 FAISS 索引中提取具体文本
    rag_results_texts = rag_module.get_texts_from_indices(rag_indices[0])
    # 打印结果
    context = " ".join(rag_results_texts)
    print('context', context)

