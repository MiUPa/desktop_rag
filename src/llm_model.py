import os
import requests

class LLMModel:
    def __init__(self, model_name=None, api_token=None):
        # ここでモデル名を変更
        if model_name is None:
            model_name = "rinna/japanese-gpt2-medium"
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # APIトークンを取得（環境変数または直接指定）
        if api_token is None:
            api_token = os.environ.get("HF_API_TOKEN", "YOUR_HF_API_TOKEN_HERE")
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    
    def generate_answer(self, query, context, max_new_tokens=150):
        # プロンプトを作成
        prompt_prefix = (
            "以下のPDF内容を参考にして、質問に対して必要な情報だけを元に、"
            "明確かつ自然な日本語で回答してください。\n\n"
        )
        input_text = (
            prompt_prefix +
            f"Context: {context}\n"
            f"Question: {query}\n"
            "Answer:"
        )
        
        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 40,
                "temperature": 0.7
            },
            "options": {
                "wait_for_model": True
            }
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"リクエストに失敗しました (ステータスコード {response.status_code}): {response.text}")
        result = response.json()
        
        if isinstance(result, list) and "generated_text" in result[0]:
            generated_text = result[0]["generated_text"]
        else:
            generated_text = ""
        
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()
        return answer