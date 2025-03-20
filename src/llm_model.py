import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import time
import traceback

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

    def retrieve_relevant_context(self, query, context, max_chunk_size=512, overlap=50, top_k=5):
        """コンテキストから質問に関連する部分を抽出する"""
        if self.use_mock or self.retrieval_model is None:
            # モックアップモードの場合は最初の500文字を返す
            return context[:min(500, len(context))]
        
        # コンテキストをチャンクに分割（より細かく）
        chunks = []
        chunk_ids = []  # ファイル名とページ番号を保持
        
        # ファイルとページごとの構造を維持
        current_file = ""
        current_page = 0
        
        lines = context.split('\n')
        current_chunk = ""
        
        for line in lines:
            if line.startswith("=== ファイル:"):
                # 新しいファイルの開始
                current_file = line.replace("=== ファイル:", "").strip()
                current_file = current_file.rstrip(" =")
                continue
            
            if line.startswith("--- ページ"):
                # 新しいページの開始
                try:
                    page_info = line.replace("--- ページ", "").strip()
                    current_page = int(page_info.split()[0])
                except:
                    current_page += 1
                
                # 前のチャンクを保存し、新しいチャンクを開始
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    chunk_ids.append(f"{current_file} (p.{current_page})")
                    current_chunk = ""
                continue
            
            current_chunk += line + "\n"
            
            # チャンクサイズをチェック
            if len(current_chunk) >= max_chunk_size:
                chunks.append(current_chunk.strip())
                chunk_ids.append(f"{current_file} (p.{current_page})")
                current_chunk = ""  # 新しいチャンクを開始
        
        # 最後のチャンクを追加
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            chunk_ids.append(f"{current_file} (p.{current_page})")
        
        if not chunks:
            print("チャンク分割後に有効なチャンクがありません。元のコンテキストの一部を返します。")
            return context[:min(1000, len(context))]
        
        try:
            # 各チャンクとクエリの埋め込みを計算
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            chunks_embeddings = self.retrieval_model.encode(chunks, convert_to_tensor=True)
            
            # コサイン類似度を計算
            similarities = util.pytorch_cos_sim(query_embedding, chunks_embeddings)[0]
            
            # 上位k個のチャンクを取得
            top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(chunks)))
            top_indices = top_k_indices.tolist()
            top_scores = top_k_values.tolist()
            
            # デバッグ情報を出力
            print(f"\n===== 関連コンテキスト検索結果 =====")
            for i, idx in enumerate(top_indices):
                print(f"スコア {top_scores[i]:.4f}: {chunk_ids[idx]}")
                print(f"プレビュー: {chunks[idx][:100]}...\n")
            
            # 関連するコンテキストを結合（ソースと類似度情報付き）
            relevant_chunks = []
            for i, idx in enumerate(top_indices):
                source_info = f"[出典: {chunk_ids[idx]}, 類似度: {top_scores[i]:.2f}]"
                chunk_with_source = f"{source_info}\n{chunks[idx]}"
                relevant_chunks.append(chunk_with_source)
            
            relevant_context = "\n\n".join(relevant_chunks)
            
            return relevant_context
        except Exception as e:
            print(f"関連コンテキスト抽出中にエラーが発生しました: {e}")
            traceback.print_exc()
            # エラーが発生した場合はコンテキストの最初の部分を返す
            return context[:min(1000, len(context))]

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
            print(f"抽出されたコンテキスト長: {len(relevant_context)} 文字")
            
            if self.use_mock or self.generation_model is None or self.tokenizer is None:
                # モックアップモードでの回答生成
                mock_answer = self.generate_mock_answer(query)
                
                # モックアップでもストリーミング風の表示を実現
                if callback:
                    for char in mock_answer:
                        callback(char)
                        time.sleep(0.03)  # モックアップの場合は文字ごとに遅延
                
                return mock_answer
            
            # 改良されたプロンプト形式
            prompt = f"""以下のコンテキストに基づいて、与えられた質問に対する回答を日本語で生成してください。
コンテキストに含まれる情報のみを使用し、含まれていない情報については「その情報はコンテキストに含まれていません」と述べてください。

コンテキスト:
{relevant_context}

質問:
{query}

指示: 
- 回答は簡潔かつ明確にしてください
- コンテキストに含まれる情報だけを使用してください
- コンテキストに情報がない場合は「その情報はコンテキストに含まれていません」と述べてください
- 出典情報（ファイル名、ページ番号）も回答に含めてください
- 統計検定の問題や内容に関する質問には、可能な限り具体的に回答してください

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
                        temperature=0.7,  # より多様性のある回答
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.92,
                        top_k=40,
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
                    
                    # 空の回答の場合はクエリに応じた代替テキストを生成
                    if not answer:
                        answer = self.generate_fallback_answer(query, relevant_context)
                    
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
                    prompt = f"""以下のコンテキスト（一部）に基づいて、与えられた質問に対する回答を日本語で生成してください。

コンテキスト（一部）:
{shorter_context}

質問:
{query}

指示: コンテキストに含まれる情報のみを使用し、含まれていない情報については「完全な情報はありません」と述べてください。

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
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_k=40,
                            repetition_penalty=1.2
                        )
                    
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    answer = generated_text[len(prompt):].strip()
                    
                    # 非ストリーミングの結果もコールバックで文字ごとに送信
                    if callback and answer:
                        for char in answer:
                            if not self.streaming_active:
                                break
                            callback(char)
                            time.sleep(0.01)
                    
                    return answer or self.generate_fallback_answer(query, shorter_context)
                else:
                    raise
        
        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            print(f"回答生成中にエラーが発生しました: {e}")
            traceback.print_exc()
            
            # エラーメッセージもストリーミングで送信
            if callback:
                for char in error_message:
                    callback(char)
                    time.sleep(0.01)
                
            return error_message

    def generate_mock_answer(self, query):
        """質問に応じたモックアップ回答を生成する"""
        if "統計検定" in query or "試験" in query:
            return f"統計検定に関するご質問「{query}」ですね。統計検定1級では確率論、統計的推測、多変量解析、時系列解析などの高度な統計学の問題が出題されます。これはモックアップモードの回答です。"
        elif "問題" in query or "例題" in query:
            return f"問題例に関するご質問「{query}」ですね。実際の問題例としては、多変量解析の固有値問題や、確率過程に関する問題などがあります。これはモックアップモードの回答です。"
        else:
            return f"ご質問「{query}」に関する回答です。現在モックアップモードで動作しているため、PDFの実際の内容に基づいた回答はできません。"

    def generate_fallback_answer(self, query, context):
        """質問に応じたフォールバック回答を生成する"""
        # コンテキストの長さを確認
        context_preview = context[:300] + "..." if len(context) > 300 else context
        
        if "統計検定" in query and "級" in query:
            level = ""
            if "1級" in query:
                level = "1級"
            elif "2級" in query:
                level = "2級"
            elif "3級" in query:
                level = "3級"
            elif "4級" in query:
                level = "4級"
            
            if level:
                return f"統計検定{level}に関する詳細な情報はコンテキストから十分に抽出できませんでした。一般的に統計検定{level}では、{'確率論、統計的推測、多変量解析、時系列解析などの高度な' if level=='1級' else '基本的な確率・統計の'}知識が問われます。"
            else:
                return f"統計検定に関する詳細情報はコンテキストから抽出できませんでした。PDFには統計検定の問題や内容に関する情報が含まれているようですが、ご質問「{query}」に対する具体的な回答を見つけることができませんでした。"
                
        elif "問題" in query or "例題" in query:
            return f"ご質問「{query}」に関する具体的な問題例はコンテキストから見つけることができませんでした。PDFには統計検定の問題が含まれていますが、該当する例題を特定できませんでした。"
            
        else:
            return f"ご質問「{query}」に対する適切な回答をコンテキストから見つけることができませんでした。PDFの内容を確認したところ、統計検定に関する情報は含まれていますが、ご質問に直接関連する情報は見つかりませんでした。より具体的な質問や、別の内容について質問してみてください。"

    def generate_answer(self, query, context):
        """質問と関連コンテキストを用いて回答を生成する（非ストリーミング）"""
        print(f"クエリ: {query}")
        
        # コールバックなしのストリーミング生成を利用
        # これにより、処理のロジックを統一できる
        answer = self.generate_answer_streaming(query, context, callback=None)
        return answer

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