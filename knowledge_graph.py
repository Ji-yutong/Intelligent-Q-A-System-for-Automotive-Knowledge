import json
import jieba
import numpy as np
from transformers import BertTokenizer, BertModel
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# 加载配置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)

class KnowledgeGraph:
    def __init__(self):
        self.wiki_api_url = "https://en.wikipedia.org/w/api.php"
        # 初始化BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(config['tiny_bert_model_path'])
        self.model = BertModel.from_pretrained(config['tiny_bert_model_path'])
        self.model.eval()

    def extract_keywords(self, context):
        # 使用jieba进行中文分词
        context = " ".join(jieba.cut(context))

        # 使用TF-IDF初步筛选关键词
        tfidf_vectorizer = TfidfVectorizer(max_features=20)             # 限制最多20个词
        tfidf_matrix = tfidf_vectorizer.fit_transform([context])        # 将上下文文本转换为TF-IDF矩阵
        feature_names = tfidf_vectorizer.get_feature_names_out()        # 获取词汇表

        # 提取TF-IDF分数最高的5个关键词
        top_features = np.array(feature_names)[np.argsort(tfidf_matrix.toarray()[0])[-5:]]
        print("Top TF-IDF Keywords:", top_features)

        # 使用BERT模型进一步处理这些关键词
        keyword_embeddings = []
        for word in top_features:
            word_inputs = self.tokenizer(word, return_tensors='pt')  # 将词转换为BERT模型输入格式
            with torch.no_grad():
                word_outputs = self.model(**word_inputs)    # 送入模型得到embedding的词
            word_embedding = word_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            keyword_embeddings.append((word, word_embedding))

        # 对句子进行BERT嵌入
        inputs = self.tokenizer(context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # 计算词嵌入和句子嵌入之间的余弦相似度
        similarities = []
        for word, word_embedding in keyword_embeddings:
            similarity = np.dot(word_embedding, sentence_embedding) / (np.linalg.norm(word_embedding) * np.linalg.norm(sentence_embedding))
            similarities.append((word, similarity))

        # 按照相似度降序排列，并取前5个最相关的词
        top_keywords = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        return [word for word, _ in top_keywords]

    def search_wikipedia(self, query):
        # 使用维基百科的 Python API 进行搜索
        # search_results = self.wiki_api.search(query, results=config['knowledge_graph_search_number'])
        search_results = wikipedia.search(query, results=config['knowledge_graph_search_number'])
        return search_results

    def enrich(self, context):
        # 提取关键词
        keywords = self.extract_keywords(context)
        print("Extracted Keywords:", keywords)

        all_results = {}
        # 对每个关键词进行维基百科搜索
        for keyword in keywords:
            search_results = self.search_wikipedia(keyword)

            if search_results:
                print(f"Search results for '{keyword}':")
                all_results[keyword] = search_results  # 将搜索结果保存到字典中
                for result in search_results:
                    print(f"- {result}")
            else:
                print(f"No results found for '{keyword}'")

        return all_results


if __name__ == "__main__":
    knowledge_graph = KnowledgeGraph()
    test_context = "美国奢侈品汽车品牌，例如法拉利、兰博基尼、保时捷等车在国际的排行榜中，谁的名气最高？" 
    print(knowledge_graph.enrich(test_context))