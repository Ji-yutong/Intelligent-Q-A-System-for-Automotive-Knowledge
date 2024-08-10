import json
import sys
import torch
from elasticsearch import Elasticsearch
from rag_module import RAGModule
from knowledge_graph import KnowledgeGraph
from multimodal_module import MultimodalModule
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import faiss
import numpy as np

# 加载配置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)


class Agent:
    def __init__(self):
        print('开始初始化各模块')
        self.dialog_history = []        # 对话历史
        self.max_dialog_length = 5      # 滑动窗口长度
        self.max_summary_length = 400   # 摘要长度限制
        self.elasticsearch = Elasticsearch([config['elasticsearch_host']])   # Elasticsearch 连接
        self.rag_module = RAGModule()   # RAG模块
        self.knowledge_graph = KnowledgeGraph()         # 知识图谱
        self.multimodal_module = MultimodalModule()     # 多模态模块
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chatglm_model = AutoModelForCausalLM.from_pretrained(config['llm_path'], trust_remote_code=True).to(self.device)# 加载 chatglm3-6b 模型
        self.chatglm_tokenizer = AutoTokenizer.from_pretrained(config['llm_path'], trust_remote_code=True)       # 加载 chatglm3-6b 分词器
        self.index = faiss.read_index(config['vector_dbindex_path'])  # FAISS 索引
        self.dimension = self.index.d        # 提取数据维度
        self.vector_data = np.load(config['vector_vectors_path'])  # 读取向量数据文件路径
        self.mt5_tokenizer = MT5Tokenizer.from_pretrained(config['summary_model_path'])    # 加载tm5分词器
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(config['summary_model_path']).to(self.device)    # 加载tm5模型
        print('各模块初始化完成')

    def summarize_dialog_history(self, dialog_history):

        # 将对话历史转换为文本
        dialog_history_text = "\n".join([f"用户: {entry.get('user_input', '')}\n系统: {entry.get('response', '')}" for entry in self.dialog_history])

        # 分词并生成摘要
        inputs = self.mt5_tokenizer("summarize: " + dialog_history_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device) if 'attention_mask' in inputs else None

        with torch.no_grad():
            outputs = self.mt5_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.max_summary_length, length_penalty=1.0,  num_beams=4, early_stopping=True)
        summary = self.mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def handle_query(self, user_input, image_path=None):

        if len(self.dialog_history) >= self.max_dialog_length:
            self.dialog_history.pop(0)  # 删除最旧的对话记录

        # 处理图片输入
        if image_path:
            image_description = self.multimodal_module.process_image(image_path)
            user_input = f"{user_input} {image_description}"  # 将图片描述加入用户输入

        # 更新对话历史
        self.dialog_history.append({'user_input': user_input})

        # 使用 Elasticsearch 搜索需要在本地安装Elasticsearch服务
        # search_result = self.elasticsearch.search(index=config['elasticsearch_index'], body={"query": {"match": {"content": user_input}}})
        # search_texts = [hit['_source']['content'] for hit in search_result['hits']['hits']]

        # RAG查询
        rag_results_texts = self.rag_module.query(user_input)[0]  # 获取 RAG 查询结果的索引
        context = " ".join(rag_results_texts)

        # 使用知识图谱对本地文档进行补充
        enriched_context = self.knowledge_graph.enrich(context)

        # 历史对话摘要
        dialog_summary = self.summarize_dialog_history(self.dialog_history)

        # print("生成prompt")
        prompt = (f"这是之前对话的总结:\n{dialog_summary}\n\n"
                  f"相关补充信息: {enriched_context}\n\n"
                  # f"搜索补充信息: {' '.join(search_texts)}\n\n"
                  f"对于这个问题: {user_input}\n\n"
                  f"有以下补充信息以供参考: {context}\n\n"
                  "请你给出问题的回复：")

        # 使用 chatglm3-6b 模型生成最终的回复
        inputs = self.chatglm_tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)        # 输入移动到gpu
        attention_mask = inputs['attention_mask'].to(self.device) if 'attention_mask' in inputs else None
        outputs = self.chatglm_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        response_text = self.chatglm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 更新模型回复历史
        self.dialog_history.append({'response': response_text})
        return response_text

    def get_dialog_history(self):
        return self.dialog_history


if __name__ == "__main__":
    agent = Agent()
    while True:
        try:
            sys.stdout.write("请输入问题：")
            sys.stdout.flush()  # 确保提示信息被打印出来
            user_input = sys.stdin.readline().strip().encode('utf-8').decode('utf-8', errors='ignore')
            if user_input.lower() == 'exit':
                break

            response = agent.handle_query(user_input)
            print("系统：", response)
        except UnicodeDecodeError as e:
            print(f"输入处理错误: {e}")
