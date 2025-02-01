from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_answer(self, query, context):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # トークン数がモデルの最大シーケンス長を超えないようにトリミング
        max_length = 1024  # GPT-2の最大シーケンス長
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]  # 最後の1024トークンを使用

        # attention_maskを設定
        attention_mask = (input_ids != self.tokenizer.pad_token_id).type(input_ids.dtype)

        # max_new_tokensを使用して新しいトークンの最大数を指定
        output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()  # "Answer:"以降の部分を返す

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
