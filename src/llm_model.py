from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LLMModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to('cpu')  # モデルをCPUに移動
        self.model.eval()  # モデルを評価モードに設定

    def generate_answer(self, query, context, max_new_tokens=150):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        # トークン化時に自動でトランケーションを適用
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.model.config.n_positions,  # 最大シーケンス長（通常は1024）
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to('cpu')
        attention_mask = encoding["attention_mask"].to('cpu')
        
        # 生成前に、入力トークン数と生成可能なトークン数の関係をチェック
        available_new_tokens = self.model.config.n_positions - input_ids.size(1)
        if available_new_tokens < max_new_tokens:
            # 生成可能なトークン数が不足している場合は、利用可能な分だけ使用
            max_new_tokens = max(1, available_new_tokens)  # 少なくとも1トークンは生成

        with torch.no_grad():  # 推論時は勾配計算を無効にする
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_p=0.95,
                top_k=50
            )
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        return answer

# モデル名を指定
model_name = "gpt2"  # または "gpt2-medium", "gpt2-large", "gpt2-xl"

# トークナイザーとモデルをダウンロード
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# モデルをローカルに保存
tokenizer.save_pretrained("./gpt2")
model.save_pretrained("./gpt2")

input_text = "あなたの質問をここに入力"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
