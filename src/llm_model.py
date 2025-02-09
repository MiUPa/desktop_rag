import os
import requests
from deepseek import DeepSeekRetrieval, DeepSeekGeneration

class LLMModel:
    def __init__(self, retrieval_model_path, generation_model_path):
        # 情報検索モデルの初期化
        self.retrieval_model = DeepSeekRetrieval.load(retrieval_model_path, trust_remote_code=True)
        # 生成モデルの初期化
        self.generation_model = DeepSeekGeneration.load(generation_model_path, trust_remote_code=True)

    def generate_answer(self, query, context):
        # 回答生成
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        answer = self.generation_model.generate(prompt)
        return answer

class RAGApp:
    def __init__(self, retrieval_model_path, generation_model_path):
        # 情報検索モデルの初期化
        self.retrieval_model = DeepSeekRetrieval.load(retrieval_model_path)
        # 生成モデルの初期化
        self.generation_model = DeepSeekGeneration.load(generation_model_path)
    
    def generate_answer(self, query):
        # 情報検索
        context = self.retrieval_model.search(query)
        # 回答生成
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        answer = self.generation_model.generate(prompt)
        return answer

# 使用例
retrieval_model_path = "/path/to/deepseek/retrieval_model"
generation_model_path = "/path/to/deepseek/generation_model"
rag_app = RAGApp(retrieval_model_path, generation_model_path)
query = "日本のAI技術について教えてください。"
answer = rag_app.generate_answer(query)
print("生成された回答:", answer)