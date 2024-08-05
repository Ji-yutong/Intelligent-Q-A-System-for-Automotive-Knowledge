import json

import torch
from elasticsearch import Elasticsearch
from rag_module import RAGModule
from knowledge_graph import KnowledgeGraph
from multimodal_module import MultimodalModule
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

# 加载配置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)

class Agent:
    def __init__(self):
        print("初始化 Elasticsearch...")
        self.elasticsearch = Elasticsearch([config['elasticsearch_host']])              # 初始化 Elasticsearch 连接
        self.dialog_history = []        # 初始化对话历史
        print("Initializing RAGModule...")
        self.rag_module = RAGModule()   # 实例化RAG模块
        print("Initializing KnowledgeGraph...")
        self.knowledge_graph = KnowledgeGraph()         # 实例化知识图谱
        print("Initializing MultimodalModule...")
        self.multimodal_module = MultimodalModule()     # 实例化多模态模块
        print("Loading chatglm3-6b model...")
        self.chatglm_model = AutoModelForCausalLM.from_pretrained(config['llm_path'], trust_remote_code=True)    # 加载 chatglm3-6b 模型
        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(config['llm_path'], trust_remote_code=True)       # 加载 chatglm3-6b 分词器
        print("Loading FAISS index...")
        self.index = faiss.read_index(config['vector_dbindex_path'])  # 加载 FAISS 索引
        self.dimension = self.index.d        # 提取数据维度
        print("Loading vector data...")
        self.vector_data = np.load(config['vector_vecotrs_path'])  # 从配置文件中读取向量数据文件路径

    def handle_query(self, user_input, image_path=None):
        print("更新对话历史")
        self.dialog_history.append({'user_input': user_input})        # 更新对话历史

        # 处理图片输入
        print("处理图片输入")
        if image_path:
            image_description = self.multimodal_module.process_image(image_path)
            user_input = f"{user_input} {image_description}"  # 将图片描述加入用户输入

        # 使用 Elasticsearch 搜索需要在本地安装Elasticsearch服务
        #search_result = self.elasticsearch.search(index=config['elasticsearch_index'], body={"query": {"match": {"content": user_input}}})
        #search_texts = [hit['_source']['content'] for hit in search_result['hits']['hits']]

        # RAG查询
        print("RAG查询开始获取 RAG 查询结果的索引")
        rag_results_indices = self.rag_module.query(user_input)[0]  # 获取 RAG 查询结果的索引
        print("RAG得到索引，开始转化成文本")
        rag_results_texts = self.rag_module.get_texts_from_indices(rag_results_indices)
        print("RAG转换文本结束，开始拼接为str")
        context = " ".join(rag_results_texts)
        enriched_context = self.knowledge_graph.enrich(context)  # 使用知识图谱增强

        # 整合对话历史
        print("整合对话历史")
        dialog_history_text = "\n".join([f"用户: {entry['user_input']}\n系统: {entry.get('response', '')}" for entry in self.dialog_history])
        # 将上下文和搜索结果整合到 prompt 中
        print("生成prompt")
        prompt = (f"之前的对话记录:\n{dialog_history_text}\n\n"
                  f"用户询问: {user_input}\n\n"
                  f"RAG匹配内容: {context}\n\n"
                  f"相关补充信息: {enriched_context}\n\n"
                  
                  "请根据以上信息给出回复：")  # f"搜索结果: {' '.join(search_texts)}\n\n"


        # 使用 chatglm3-6b 模型生成最终的回复
        #device = torch.device('cuda')
        #self.chatglm_model.to(device)
        print("prompt送入模型tokenizer，开始对prompt进行编码")
        inputs = self.chatglm_tokenizer(prompt, return_tensors='pt')
        #inputs = {key: value.to(device) for key, value in inputs.items()}
        print("将编码的prompt送入模型，开始生成回复")
        outputs = self.chatglm_model.generate(**inputs, max_length=512, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        print("得到向量后转换成文本")
        response_text = self.chatglm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("更新对话历史记录")
        self.dialog_history.append({'response': response_text})  # 更新模型回复历史
        print("功能完成")
        return response_text

    def get_dialog_history(self):
        return self.dialog_history


if __name__ == "__main__":
    agent = Agent()
    test_query = "汽车轮胎漏气时应该如何处理？"
    response = agent.handle_query(test_query)
    print("Response:", json.dumps(response, indent=2))
    print("Dialog History:", json.dumps(agent.get_dialog_history(), indent=2))
