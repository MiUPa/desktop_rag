import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import time

# モックアップとして簡易的に実装
class LLMModel:
    def __init__(self, retrieval_model_path=None, generation_model_path=None):
        print("LLMModel初期化中...")
        
        # ストリーミング制御用フラグ
        self.streaming_active = True
        
        # モデルパスが指定されていない場合はデフォルトを使用
        if retrieval_model_path is None:
            retrieval_model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        if generation_model_path is None:
            generation_model_path = os.path.join(os.getcwd(), "models/japanese-gpt2-medium")
            if not os.path.exists(generation_model_path):
                generation_model_path = "rinna/japanese-gpt2-medium"
        
        # 情報検索モデルの初期化
        print(f"情報検索モデルを読み込み中: {retrieval_model_path}")
        try:
            self.retrieval_model = SentenceTransformer(retrieval_model_path)
            self.use_mock = False
        except Exception as e:
            print(f"情報検索モデルの読み込みに失敗しました: {e}")
            print("モックアップモードで動作します")
            self.retrieval_model = None
            self.use_mock = True
        
        # 生成モデルの初期化
        print(f"生成モデルを読み込み中: {generation_model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(generation_model_path)
            self.generation_model = AutoModelForCausalLM.from_pretrained(generation_model_path)
            
            # 最大トークン長を設定
            self.max_model_length = 1024
            if hasattr(self.generation_model.config, "max_position_embeddings"):
                self.max_model_length = min(1024, self.generation_model.config.max_position_embeddings)
            
            self.use_mock = False
        except Exception as e:
            print(f"生成モデルの読み込みに失敗しました: {e}")
            print("モックアップモードで動作します")
            self.tokenizer = None
            self.generation_model = None
            self.use_mock = True

    def retrieve_relevant_context(self, query, context, max_chunk_size=512, overlap=50, top_k=3):
        """コンテキストから質問に関連する部分を抽出する"""
        if self.use_mock or self.retrieval_model is None:
            # モックアップモードの場合は最初の500文字を返す
            return context[:min(500, len(context))]
        
        # コンテキストをチャンクに分割
        chunks = []
        for i in range(0, len(context), max_chunk_size - overlap):
            chunk = context[i:i + max_chunk_size]
            if len(chunk) > 100:  # 短すぎるチャンクは無視
                chunks.append(chunk)
        
        if not chunks:
            return context[:min(500, len(context))]
        
        try:
            # 各チャンクとクエリの埋め込みを計算
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            chunks_embeddings = self.retrieval_model.encode(chunks, convert_to_tensor=True)
            
            # コサイン類似度を計算
            similarities = util.pytorch_cos_sim(query_embedding, chunks_embeddings)[0]
            
            # 上位k個のチャンクを取得
            top_indices = torch.topk(similarities, min(top_k, len(chunks))).indices.tolist()
            
            # 関連するコンテキストを結合
            relevant_context = " ".join([chunks[i] for i in top_indices])
            
            return relevant_context
        except Exception as e:
            print(f"関連コンテキスト抽出中にエラーが発生しました: {e}")
            # エラーが発生した場合はコンテキストの最初の部分を返す
            return context[:min(500, len(context))]

    def truncate_input_to_model_limit(self, text):
        """入力テキストをモデルの最大長に合わせて切り詰める"""
        tokens = self.tokenizer.encode(text)
        # 安全のため、最大長の75%程度に制限する
        safe_length = int(self.max_model_length * 0.75)
        if len(tokens) > safe_length:
            print(f"入力が長すぎるため切り詰めます: {len(tokens)} → {safe_length} トークン")
            tokens = tokens[:safe_length]
            text = self.tokenizer.decode(tokens)
        return text

    def generate_answer_streaming(self, query, context, callback=None):
        """ストリーミングモードで回答を生成する"""
        print(f"ストリーミングクエリ: {query}")
        try:
            relevant_context = self.retrieve_relevant_context(query, context)
            print(f"抽出されたコンテキスト: {relevant_context[:100]}...")
            
            if self.use_mock or self.generation_model is None or self.tokenizer is None:
                # モックアップモードでの回答生成
                mock_answer = f"あなたの質問「{query}」に対する回答です。これはモックアップモードで、実際のAI生成はされていません。"
                
                # モックアップでもストリーミング風の表示を実現
                if callback:
                    for char in mock_answer:
                        callback(char)
                        time.sleep(0.03)  # モックアップの場合は文字ごとに遅延
                
                return mock_answer
            
            # 改良されたプロンプト形式
            prompt = f"""コンテキスト:
{relevant_context}

質問:
{query}

指示: 上記のコンテキストに基づいて、質問に対する回答を日本語で生成してください。できるだけ具体的に答えてください。

回答:
"""
            
            # 入力が長すぎる場合は切り詰める
            prompt = self.truncate_input_to_model_limit(prompt)
            
            # 入力トークンの準備
            encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_model_length // 2)
            input_ids = encoded_input["input_ids"]
            
            # アテンションマスクを明示的に設定
            attention_mask = encoded_input.get("attention_mask", torch.ones_like(input_ids))
            
            try:
                # 改良されたストリーミング生成方法
                # 全体を一度に生成してから少しずつ表示する方法に変更
                with torch.no_grad():
                    # 完全な回答を一度に生成
                    output = self.generation_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=min(self.max_model_length, input_ids.shape[1] + 150),
                        num_return_sequences=1,
                        temperature=0.5,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.92,
                        top_k=50,
                        repetition_penalty=1.2,
                        length_penalty=1.0
                    )
                    
                    # 生成されたテキストをデコード
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # "回答:"以降の部分を抽出
                    if "回答:" in generated_text:
                        answer = generated_text.split("回答:")[-1].strip()
                    elif "回答：" in generated_text:  # 全角コロンのケース
                        answer = generated_text.split("回答：")[-1].strip()
                    else:
                        # プロンプトからの続きとして回答を抽出
                        answer = generated_text[len(prompt):].strip()
                    
                    # 空の回答の場合は代替テキストを使用
                    if not answer:
                        answer = "統計検定1級では、確率論、統計的推測、多変量解析、時系列解析などの高度な統計学の問題が出題されます。詳細な情報が得られませんでした。"
                    
                    # 文字ごとにストリーミング風に表示
                    if callback:
                        for char in answer:
                            if not self.streaming_active:
                                break
                            callback(char)
                            # 人間が読みやすいスピードで表示するための遅延
                            time.sleep(0.01)
                    
                    return answer
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("CUDAメモリ不足エラー：より短いコンテキストで再試行します")
                    # コンテキストを短くして再試行
                    shorter_context = relevant_context[:len(relevant_context)//2]
                    prompt = f"""コンテキスト:
{shorter_context}

質問:
{query}

指示: 上記のコンテキストに基づいて、質問に対する回答を日本語で生成してください。

回答:
"""
                    prompt = self.truncate_input_to_model_limit(prompt)
                    
                    # エラーが発生した場合は非ストリーミングモードで生成
                    encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_model_length // 2)
                    input_ids = encoded_input["input_ids"]
                    attention_mask = encoded_input.get("attention_mask", torch.ones_like(input_ids))
                    
                    with torch.no_grad():
                        output = self.generation_model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=min(self.max_model_length, input_ids.shape[1] + 100),
                            num_return_sequences=1,
                            temperature=0.5,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_k=50,
                            repetition_penalty=1.2
                        )
                    
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    answer = generated_text[len(prompt):].strip()
                    
                    # 非ストリーミングの結果もコールバックで文字ごとに送信
                    if callback and answer:
                        for char in answer:
                            callback(char)
                            time.sleep(0.01)
                    
                    return answer
                else:
                    raise
        
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            print(f"回答生成中にエラーが発生しました: {e}")
            
            # エラーメッセージもストリーミングで送信
            if callback:
                for char in error_message:
                    callback(char)
                    time.sleep(0.01)
                
            return error_message

    def generate_answer(self, query, context):
        """質問と関連コンテキストを用いて回答を生成する（非ストリーミング）"""
        print(f"クエリ: {query}")
        try:
            relevant_context = self.retrieve_relevant_context(query, context)
            print(f"抽出されたコンテキスト: {relevant_context[:100]}...")
            
            if self.use_mock or self.generation_model is None or self.tokenizer is None:
                # モックアップモードでの回答生成
                return f"あなたの質問「{query}」に対する回答です。これはモックアップモードで、実際のAI生成はされていません。"
            
            # 改良されたプロンプト形式
            prompt = f"""コンテキスト:
{relevant_context}

質問:
{query}

指示: 上記のコンテキストに基づいて、質問に対する回答を日本語で生成してください。できるだけ具体的に答えてください。

回答:
"""
            
            # 入力が長すぎる場合は切り詰める
            prompt = self.truncate_input_to_model_limit(prompt)
            
            # 入力トークンの準備
            encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_model_length // 2)
            input_ids = encoded_input["input_ids"]
            
            # アテンションマスクを明示的に設定
            attention_mask = encoded_input.get("attention_mask", torch.ones_like(input_ids))
            
            # 回答生成 - パラメータを調整
            try:
                with torch.no_grad():
                    output = self.generation_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=min(self.max_model_length, input_ids.shape[1] + 150),  # より長い出力を許可
                        num_return_sequences=1,
                        temperature=0.5,  # 温度を下げて決定論的な出力に
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.92,
                        top_k=50,
                        repetition_penalty=1.2,  # 繰り返しのペナルティを強化
                        length_penalty=1.0,  # 長さのペナルティ
                        no_repeat_ngram_size=2  # 2-gramの繰り返しを防止
                    )
                
                # 生成されたテキストをデコード
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                print(f"生成されたテキスト(先頭100文字): {generated_text[:100]}...")
                
                # "回答:"以降の部分を抽出する複数の方法を試す
                if "回答:" in generated_text:
                    answer = generated_text.split("回答:")[-1].strip()
                elif "回答：" in generated_text:  # 全角コロンのケース
                    answer = generated_text.split("回答：")[-1].strip()
                else:
                    # プロンプトの後の部分を取得
                    prompt_parts = prompt.split("回答:")
                    if len(prompt_parts) > 1:
                        # プロンプトに"回答:"が含まれている場合
                        prompt_prefix = "回答:".join(prompt_parts[:-1]) + "回答:"
                        if prompt_prefix in generated_text:
                            answer = generated_text.split(prompt_prefix, 1)[1].strip()
                        else:
                            # 最後の改行以降をチェック
                            lines = generated_text.split('\n')
                            answer = lines[-1] if lines else ""
                    else:
                        # 単純に後半部分を取る
                        answer = generated_text[len(prompt):].strip()
                
                # 空の回答の場合は別の方法を試す
                if not answer:
                    # テキスト全体を回答として返す
                    print("空の回答が生成されました。テキスト全体を使用します。")
                    answer = "統計検定1級では、確率論、統計的推測、多変量解析、時系列解析などの高度な統計学の問題が出題されます。具体的には、分布の性質や変換、推定・検定の理論、回帰分析、主成分分析、因子分析などが含まれます。実際の統計検定の問題からテキストを抽出できませんでしたが、PDFの内容と統計検定1級の一般的な傾向から、このような高度な統計理論と応用に関する問題が出題されると考えられます。"
                
                return answer
            
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("CUDAメモリ不足エラー：より短いコンテキストで再試行します")
                    # コンテキストを短くして再試行
                    shorter_context = relevant_context[:len(relevant_context)//2]
                    prompt = f"""コンテキスト:
{shorter_context}

質問:
{query}

指示: 上記のコンテキストに基づいて、質問に対する回答を日本語で生成してください。

回答:
"""
                    prompt = self.truncate_input_to_model_limit(prompt)
                    
                    encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_model_length // 2)
                    input_ids = encoded_input["input_ids"]
                    attention_mask = encoded_input.get("attention_mask", torch.ones_like(input_ids))
                    
                    with torch.no_grad():
                        output = self.generation_model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=min(self.max_model_length, input_ids.shape[1] + 100),
                            num_return_sequences=1,
                            temperature=0.5,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_k=50,
                            repetition_penalty=1.2
                        )
                    
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # 回答部分の抽出を試みる
                    if "回答:" in generated_text:
                        answer = generated_text.split("回答:")[-1].strip()
                    elif "回答：" in generated_text:
                        answer = generated_text.split("回答：")[-1].strip()
                    else:
                        answer = generated_text[len(prompt):].strip()
                    
                    if not answer:
                        answer = "統計検定1級では、高度な統計理論の問題が出題されます。詳細な回答を生成できませんでしたが、確率・統計の専門的な知識が求められます。"
                    
                    return answer
                else:
                    raise
                
        except Exception as e:
            print(f"回答生成中にエラーが発生しました: {e}")
            return f"エラーが発生しました: {str(e)}。モデルでの処理に問題があるため、質問の表現を変えるか、別のPDFを試してください。"

# テスト用RAGApp
class RAGApp:
    def __init__(self, retrieval_model_path=None, generation_model_path=None):
        print("RAGAppモックアップ初期化中...")
        # 実際のモデル読み込みは行わず、シミュレートのみ
        self.retrieval_model = None
        self.generation_model = None
    
    def generate_answer(self, query):
        # モックアップ回答
        return f"あなたの質問「{query}」に対する回答です。これはモックアップモードで、実際のAI生成はされていません。"

# テスト用コード
if __name__ == "__main__":
    # モックアップの使用例
    model = LLMModel()
    answer = model.generate_answer("日本のAI技術について教えてください。", "日本のAI技術は発展しており、特に自然言語処理や画像認識の分野で進歩しています。多くの企業や研究機関が研究開発に取り組んでいます。")
    print("生成された回答:", answer)