import os
os.environ["TRANSFORMERS_NO_SAFE_TENSORS"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMModel:
    def __init__(self, model_name=None):
        if model_name is None:
            # 現在のスクリプトの絶対パスから、正しい絶対パスに変換する
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_name = os.path.abspath(os.path.join(base_dir, "..", "models", "japanese-gpt2-medium"))
        
        # READMEの指示通りにAutoTokenizerを使用
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,  # READMEの指示通り
            use_local_files_only=True
        )
        # do_lower_caseの設定は不要です。すでにトークナイザー内で正しく設定されています。
        
        # モデルの読み込み
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True
        )
        self.model.to('cpu')
        self.model.eval()
        
    def generate_answer(self, query, context, max_new_tokens=150):
        prompt_prefix = "以下のPDF内容を参考にして、質問に対して必要な情報だけを元に、自然な日本語で回答してください。\n\n"
        input_text = prompt_prefix + f"Context: {context}\nQuestion: {query}\nAnswer:"
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.model.config.n_positions,
            return_tensors="pt",
            padding=True
        )
        
        input_ids = encoding["input_ids"].to('cpu')
        attention_mask = encoding["attention_mask"].to('cpu')
        
        available_new_tokens = self.model.config.n_positions - input_ids.size(1)
        if available_new_tokens < max_new_tokens:
            max_new_tokens = max(1, available_new_tokens)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                temperature=0.7
            )
            
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        return answer