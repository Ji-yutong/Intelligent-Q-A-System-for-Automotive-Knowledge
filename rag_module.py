import os
import faiss
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from sentence_transformers import CrossEncoder


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 加载配置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)


class RAGModule:
    def __init__(self):
        # embedding模型
        self.tokenizer = AutoTokenizer.from_pretrained(config['embedding_model_path'])
        self.model = AutoModel.from_pretrained(config['embedding_model_path'])
        self.index = faiss.read_index(config['vector_dbindex_path'])  # 加载FAISS索引文件
        self.top_k = config['rag_retrieval_top_k']                    # 加载top_k

        # 加载 roberta-base 模型
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta-base_path'])
        self.roberta_model = RobertaModel.from_pretrained(config['roberta-base_path'])

        # 加载 cross-encoder 模型
        self.cross_encoder = CrossEncoder(config['cross-encoder_path'])

        # 加载文本块与索引的映射
        with open(config['text_mapping_path'], 'r', encoding='utf-8') as f:
            self.text_mapping = json.load(f)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] token 的输出作为句子向量
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def roberta_encode(self, texts):
        inputs = self.roberta_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def query(self, query_text):
        # 使用 FAISS 进行初步召回
        query_vector = self.encode_text(query_text)
        distances, indices = self.index.search(query_vector, self.top_k)  # 召回 top_k 个文本块

        # 获取初步召回的文本块
        candidate_texts = [self.text_mapping[str(idx)] for idx in indices[0]]

        # 使用 roberta-base 进行高级召回
        roberta_query_vector = self.roberta_encode([query_text])[0]
        roberta_text_vectors = self.roberta_encode(candidate_texts)
        similarities = np.dot(roberta_text_vectors, roberta_query_vector) / (
                    np.linalg.norm(roberta_text_vectors, axis=1) * np.linalg.norm(roberta_query_vector))

        # 获取与查询文本最相关的 5 个文本块
        top_indices = np.argsort(similarities)[-5:]
        refined_texts = [candidate_texts[i] for i in top_indices]

        # 使用 cross-encoder 进行重排序
        cross_encoder_scores = self.cross_encoder.predict([(query_text, text) for text in refined_texts])
        ranked_texts = [text for _, text in
                        sorted(zip(cross_encoder_scores, refined_texts), key=lambda x: x[0], reverse=True)]

        return ranked_texts


if __name__ == "__main__":
    rag_module = RAGModule()
    test_query = "汽车轮胎漏气时应该如何处理？"
    rag_results_texts = rag_module.query(test_query)
    print('context', rag_results_texts)

