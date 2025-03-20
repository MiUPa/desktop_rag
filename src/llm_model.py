import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import time
import traceback
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# モックアップとして簡易的に実装
class LLMModel:
    def __init__(self, retrieval_model_path, generation_model_path):
        print("LLMModel初期化中...")
        self.streaming_active = True
        
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
        
        # 生成モデルの選択
        # 指定されたモデルをそのまま使用（自動切り替えを無効化）
        print(f"文章生成モデルを読み込み中: {generation_model_path}")
        
        try:
            # トークナイザの設定
            self.tokenizer = AutoTokenizer.from_pretrained(
                generation_model_path,
                use_fast=False  # 一部の日本語モデルではFast Tokenizerに問題がある場合がある
            )
            
            # モデルの設定とロード
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                generation_model_path, 
                torch_dtype=torch.float32
            )
            
            # 最大トークン長を設定
            self.max_model_length = 1024  # GPT-2モデル用に1024に設定
            if hasattr(self.generation_model.config, "max_position_embeddings"):
                self.max_model_length = min(1024, getattr(self.generation_model.config, "max_position_embeddings", 1024))
                print(f"モデルの最大コンテキスト長: {self.max_model_length} トークン")
            
            self.use_mock = False
        except Exception as e:
            print(f"生成モデルの読み込みに失敗しました: {e}")
            print("モックアップモードで動作します")
            self.tokenizer = None
            self.generation_model = None
            self.use_mock = True

    def retrieve_relevant_context(self, query, context, max_chunk_size=1024, overlap=200, top_k=5):
        """ハイブリッド検索によりコンテキストから質問に関連する部分を抽出する
        階層的チャンキングを使用: 検索には小さなチャンクを使いつつ、返却時には文脈を保持した大きなチャンクを返す
        """
        if self.use_mock or self.retrieval_model is None:
            # モックアップモードの場合は最初の1000文字を返す
            return context[:min(1000, len(context))]
        
        # 階層的チャンキング:
        # 1. 検索用の小さなチャンク (search_chunks)
        # 2. 返却用の大きなチャンク (retrieval_chunks)
        search_chunks = []       # 検索用の小さなチャンク
        retrieval_chunks = []    # 返却用の大きなチャンク
        search_chunk_ids = []    # 検索用チャンクのID
        retrieval_chunk_ids = [] # 返却用チャンクのID
        search_to_retrieval_map = {}  # 検索用チャンクから返却用チャンクへのマッピング
        
        # 検索用の小さなチャンクサイズと返却用の大きなチャンクサイズ
        search_chunk_size = max_chunk_size // 2  # 検索用は小さめ (512文字程度)
        retrieval_chunk_size = max_chunk_size * 2  # 返却用は大きめ (2048文字程度)
        
        # ファイルとページごとの構造を維持
        current_file = ""
        current_page = 0
        
        lines = context.split('\n')
        current_search_chunk = ""
        current_retrieval_chunk = ""
        
        search_chunk_index = 0
        retrieval_chunk_index = 0
        
        for line in lines:
            if line.startswith("=== ファイル:"):
                # 新しいファイルの開始
                current_file = line.replace("=== ファイル:", "").strip()
                current_file = current_file.rstrip(" =")
                
                # 既存のチャンクを保存
                if current_search_chunk.strip():
                    search_chunks.append(current_search_chunk.strip())
                    search_chunk_ids.append(f"{current_file} (p.{current_page})")
                    search_chunk_index += 1
                    current_search_chunk = ""
                
                if current_retrieval_chunk.strip():
                    retrieval_chunks.append(current_retrieval_chunk.strip())
                    retrieval_chunk_ids.append(f"{current_file} (p.{current_page})")
                    retrieval_chunk_index += 1
                    current_retrieval_chunk = ""
                
                # ファイル行自体は両方のチャンクに追加
                current_search_chunk += line + "\n"
                current_retrieval_chunk += line + "\n"
                continue
            
            if line.startswith("--- ページ"):
                # 新しいページの開始
                try:
                    page_info = line.replace("--- ページ", "").strip()
                    current_page = int(page_info.split()[0])
                except:
                    current_page += 1
                
                # ページ行自体は両方のチャンクに追加
                current_search_chunk += line + "\n"
                current_retrieval_chunk += line + "\n"
                
                # 検索用チャンクがサイズを超えていたら保存
                if len(current_search_chunk) >= search_chunk_size:
                    search_chunks.append(current_search_chunk.strip())
                    search_chunk_ids.append(f"{current_file} (p.{current_page})")
                    # このインデックスが属する返却用チャンクのインデックスをマッピング
                    search_to_retrieval_map[search_chunk_index] = retrieval_chunk_index
                    search_chunk_index += 1
                    current_search_chunk = ""
                
                # 返却用チャンクがサイズを超えていたら保存
                if len(current_retrieval_chunk) >= retrieval_chunk_size:
                    retrieval_chunks.append(current_retrieval_chunk.strip())
                    retrieval_chunk_ids.append(f"{current_file} (p.{current_page})")
                    retrieval_chunk_index += 1
                    current_retrieval_chunk = ""
                
                continue
            
            # 通常のテキスト行
            current_search_chunk += line + "\n"
            current_retrieval_chunk += line + "\n"
            
            # 検索用チャンクがサイズを超えていたら保存
            if len(current_search_chunk) >= search_chunk_size:
                search_chunks.append(current_search_chunk.strip())
                search_chunk_ids.append(f"{current_file} (p.{current_page})")
                # このインデックスが属する返却用チャンクのインデックスをマッピング
                search_to_retrieval_map[search_chunk_index] = retrieval_chunk_index
                search_chunk_index += 1
                
                # 検索チャンクはオーバーラップありで初期化
                current_search_chunk = current_search_chunk[-overlap:] if overlap > 0 else ""
            
            # 返却用チャンクがサイズを超えていたら保存
            if len(current_retrieval_chunk) >= retrieval_chunk_size:
                retrieval_chunks.append(current_retrieval_chunk.strip())
                retrieval_chunk_ids.append(f"{current_file} (p.{current_page})")
                retrieval_chunk_index += 1
                
                # 返却用チャンクもオーバーラップありで初期化 (ただし少なめ)
                retrieval_overlap = min(overlap * 2, len(current_retrieval_chunk) // 4)
                current_retrieval_chunk = current_retrieval_chunk[-retrieval_overlap:] if retrieval_overlap > 0 else ""
        
        # 最後のチャンクを追加
        if current_search_chunk.strip():
            search_chunks.append(current_search_chunk.strip())
            search_chunk_ids.append(f"{current_file} (p.{current_page})")
            # このインデックスが属する返却用チャンクのインデックスをマッピング
            search_to_retrieval_map[search_chunk_index] = retrieval_chunk_index
        
        if current_retrieval_chunk.strip():
            retrieval_chunks.append(current_retrieval_chunk.strip())
            retrieval_chunk_ids.append(f"{current_file} (p.{current_page})")
        
        if not search_chunks or not retrieval_chunks:
            print("チャンク分割後に有効なチャンクがありません。元のコンテキストの一部を返します。")
            return context[:min(2000, len(context))]
        
        try:
            # 1. セマンティック検索（ベクトル類似度）- 検索用チャンクに対して
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            search_chunks_embeddings = self.retrieval_model.encode(search_chunks, convert_to_tensor=True)
            
            semantic_similarities = util.pytorch_cos_sim(query_embedding, search_chunks_embeddings)[0]
            semantic_scores = semantic_similarities.cpu().numpy()
            
            # 2. キーワード検索（BM25）- 検索用チャンクに対して
            # クエリの前処理
            processed_query = self.preprocess_text(query)
            processed_chunks = [self.preprocess_text(chunk) for chunk in search_chunks]
            
            # BM25の実装
            tokenized_chunks = [chunk.split() for chunk in processed_chunks]
            bm25 = BM25Okapi(tokenized_chunks)
            tokenized_query = processed_query.split()
            bm25_scores = np.array(bm25.get_scores(tokenized_query))
            
            # BM25スコアを正規化（0-1の範囲に）
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            
            # 3. ハイブリッドスコアの計算（重み付け）
            semantic_weight = 0.7  # セマンティック検索の重み
            keyword_weight = 0.3   # キーワード検索の重み
            
            # ハイブリッドスコアの計算
            hybrid_scores = (semantic_weight * semantic_scores) + (keyword_weight * bm25_scores)
            
            # 上位k個の検索用チャンクを取得
            top_indices = np.argsort(hybrid_scores)[::-1][:min(top_k*2, len(search_chunks))].tolist()
            top_scores = [hybrid_scores[i] for i in top_indices]
            
            # クエリ拡張（クエリに近い単語を追加）
            expanded_query = self.expand_query(query, processed_chunks)
            print(f"拡張クエリ: {expanded_query}")
            
            # デバッグ情報を出力
            print(f"\n===== ハイブリッド検索結果 =====")
            for i, idx in enumerate(top_indices[:min(5, len(top_indices))]):  # 最大5つまで表示
                semantic_score = semantic_scores[idx]
                keyword_score = bm25_scores[idx]
                hybrid_score = hybrid_scores[idx]
                
                print(f"検索チャンク {idx}: {search_chunk_ids[idx]}")
                print(f"  セマンティックスコア: {semantic_score:.4f}, キーワードスコア: {keyword_score:.4f}, ハイブリッドスコア: {hybrid_score:.4f}")
                print(f"  プレビュー: {search_chunks[idx][:100]}...\n")
            
            # スコアが低すぎるチャンクを除外（閾値を設定）
            threshold = 0.15  # 閾値を低めに設定
            filtered_indices = [idx for i, idx in enumerate(top_indices) if top_scores[i] >= threshold]
            
            if not filtered_indices:
                print(f"関連性の高いチャンクが見つかりませんでした（閾値: {threshold}）")
                # スコアに関わらず上位3つを使用
                filtered_indices = top_indices[:min(3, len(top_indices))]
            
            # 検索用チャンクから対応する返却用チャンクへのマッピング
            # 各検索チャンクが属する返却用チャンクのインデックスを見つける
            retrieval_indices = set()
            for search_idx in filtered_indices:
                # 各検索用チャンクに対応する返却用チャンクのインデックスを取得
                # 直接マッピングがなければ最も近い返却用チャンクを探す
                if search_idx in search_to_retrieval_map:
                    retrieval_idx = search_to_retrieval_map[search_idx]
                else:
                    # 近似的な対応付け: 検索チャンクインデックス / 検索チャンク総数 * 返却チャンク総数
                    ratio = search_idx / max(1, len(search_chunks))
                    retrieval_idx = min(int(ratio * len(retrieval_chunks)), len(retrieval_chunks) - 1)
                
                retrieval_indices.add(retrieval_idx)
            
            # 返却用チャンクをスコア順にソートして結合
            # 各検索用チャンクのスコアを対応する返却用チャンクに転送
            retrieval_scores = {}
            for search_idx in filtered_indices:
                if search_idx in search_to_retrieval_map:
                    retrieval_idx = search_to_retrieval_map[search_idx]
                else:
                    ratio = search_idx / max(1, len(search_chunks))
                    retrieval_idx = min(int(ratio * len(retrieval_chunks)), len(retrieval_chunks) - 1)
                
                score = hybrid_scores[search_idx]
                # 同じ返却用チャンクに複数の検索用チャンクが対応する場合は最大スコアを使用
                retrieval_scores[retrieval_idx] = max(retrieval_scores.get(retrieval_idx, 0), score)
            
            # 返却用チャンクをスコア順にソート
            sorted_retrieval_indices = sorted(
                list(retrieval_indices), 
                key=lambda idx: retrieval_scores.get(idx, 0), 
                reverse=True
            )
            
            # 上位の返却用チャンクだけを使用
            retrieval_top_k = min(top_k, len(sorted_retrieval_indices))
            sorted_retrieval_indices = sorted_retrieval_indices[:retrieval_top_k]
            
            # 関連するコンテキストを結合（ソースと類似度情報付き）
            relevant_chunks = []
            
            print("\n===== 選択された返却チャンク =====")
            for i, idx in enumerate(sorted_retrieval_indices):
                score = retrieval_scores.get(idx, 0)
                source_info = f"[出典: {retrieval_chunk_ids[idx]}, 関連スコア: {score:.2f}]"
                print(f"返却チャンク {idx}: {retrieval_chunk_ids[idx]}, スコア: {score:.4f}")
                print(f"  プレビュー: {retrieval_chunks[idx][:100]}...\n")
                
                chunk_with_source = f"{source_info}\n{retrieval_chunks[idx]}"
                relevant_chunks.append(chunk_with_source)
            
            # 返却用チャンクを結合
            relevant_context = "\n\n".join(relevant_chunks)
            
            # 文脈を分析して改善
            if len(query.split()) >= 3:  # クエリが十分に長い場合
                # クエリ特有の語句を強調
                query_terms = set(self.preprocess_text(query).split())
                for term in query_terms:
                    if len(term) > 1 and term not in ['は', 'の', 'が', 'を', 'に', 'へ', 'と', 'より', 'から']:
                        # 重要な語句がある行を強調（重複排除して実行）
                        relevant_context = re.sub(
                            f'([^。]*{term}[^。]*。)',
                            r'【重要】\1',
                            relevant_context
                        )
            
            return relevant_context
        except Exception as e:
            print(f"関連コンテキスト抽出中にエラーが発生しました: {e}")
            traceback.print_exc()
            # エラーが発生した場合はコンテキストの最初の部分を返す
            return context[:min(2000, len(context))]
    
    def preprocess_text(self, text):
        """テキストの前処理：小文字化、不要な文字の削除など"""
        # 日本語テキストの前処理
        text = text.lower()
        # 特殊文字を削除
        text = re.sub(r'[「」『』（）\(\)\[\]\{\}]', ' ', text)
        # 複数のスペースを1つに
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def expand_query(self, query, corpus, top_n=3):
        """TF-IDFを使用してクエリを拡張する"""
        try:
            # Vectorizerを初期化
            vectorizer = TfidfVectorizer(max_features=1000)
            
            # コーパスにクエリを追加
            all_texts = corpus + [query]
            
            # TF-IDF行列を作成
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # クエリのTF-IDFベクトル（最後の要素）
            query_vector = tfidf_matrix[-1]
            
            # コーパスのTF-IDFベクトル
            corpus_tfidf = tfidf_matrix[:-1]
            
            # クエリと各文書の類似度を計算
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, corpus_tfidf)[0]
            
            # 最も類似度が高い文書のインデックス
            top_doc_indices = similarities.argsort()[-top_n:][::-1]
            
            # 拡張語を取得
            feature_names = vectorizer.get_feature_names_out()
            query_terms = set(self.preprocess_text(query).split())
            expansion_terms = set()
            
            for doc_idx in top_doc_indices:
                doc_vector = corpus_tfidf[doc_idx]
                # この文書での重要な単語のインデックスを取得
                word_indices = doc_vector.toarray()[0].argsort()[-5:][::-1]
                for word_idx in word_indices:
                    term = feature_names[word_idx]
                    if term not in query_terms and len(term) > 1:
                        expansion_terms.add(term)
            
            # 元のクエリと拡張語を組み合わせる
            expanded_query = query + " " + " ".join(list(expansion_terms)[:3])
            return expanded_query
        except Exception as e:
            print(f"クエリ拡張中にエラーが発生しました: {e}")
            return query  # エラーの場合は元のクエリを返す

    def truncate_input_to_model_limit(self, text):
        """入力テキストがモデルの制限を超えないように切り詰める"""
        if self.tokenizer is None:
            return text
        
        # テキストをトークン化してトークン数をカウント
        tokens = self.tokenizer.encode(text)
        
        # 許容トークン数を小さなモデルに合わせて調整（最大長の60%に設定）
        max_allowed_tokens = int(self.max_model_length * 0.6)
        
        if len(tokens) <= max_allowed_tokens:
            return text
        
        # トークン数が制限を超える場合は切り詰める
        truncated_tokens = tokens[:max_allowed_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        print(f"テキストが長すぎるため切り詰めました: {len(tokens)} → {len(truncated_tokens)} トークン")
        
        return truncated_text

    def generate_answer(self, query, context):
        """質問と関連コンテキストを用いて回答を生成する（非ストリーミング）"""
        print(f"クエリ: {query}")
        
        # コールバックなしのストリーミング生成を利用
        # これにより、処理のロジックを統一できる
        answer = self.generate_answer_streaming(query, context, callback=None)
        return answer

    def generate_answer_streaming(self, query, context, callback=None, conversation_history=""):
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
            
            # コンテキストが空または非常に短い場合は代替テキストを生成
            if len(relevant_context.strip()) < 50:
                fallback = self.generate_fallback_answer(query, context[:1000])
                if callback:
                    for char in fallback:
                        callback(char)
                        time.sleep(0.01)
                return fallback
            
            # 会話履歴を含む改良されたプロンプト形式
            # 会話履歴があれば追加
            history_prefix = ""
            if conversation_history:
                history_prefix = f"{conversation_history}\n"
            
            # 改良されたプロンプト：より具体的な指示と制約を含む
            prompt = f"""{history_prefix}あなたは優秀な日本語AIアシスタントです。以下の情報源だけを使用して質問に答えてください。

【情報源】
{relevant_context}

【質問】
{query}

【回答の作成指示】
1. 情報源に明示的に含まれている情報だけを使用して回答してください
2. 情報源に情報がない場合は、「申し訳ありませんが、提供された情報源にはその質問に対する回答が含まれていません」と正直に述べてください
3. 情報を捏造したり、推測したりしないでください
4. 回答にはPDFの出典情報（ファイル名、ページ番号）を含めてください
5. 回答は論理的で、事実に基づき、具体的であるべきです
6. 会話の流れを考慮して、自然な応答を心がけてください

【回答】
"""
            
            # 入力が長すぎる場合は切り詰める
            prompt = self.truncate_input_to_model_limit(prompt)
            
            # 入力トークンの準備
            # より大きなmax_lengthを設定（モデルの最大長の75%まで）
            max_length = int(self.max_model_length * 0.75)
            encoded_input = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encoded_input["input_ids"].to(self.generation_model.device)
            
            # アテンションマスクを明示的に設定
            attention_mask = encoded_input.get("attention_mask", torch.ones_like(input_ids)).to(self.generation_model.device)
            
            try:
                # 改良されたストリーミング生成方法
                # 全体を一度に生成してから少しずつ表示する方法に変更
                with torch.no_grad():
                    # 完全な回答を一度に生成
                    # 大きなモデルでは生成パラメータを調整
                    output = self.generation_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=min(self.max_model_length, input_ids.shape[1] + 200),
                        num_return_sequences=1,
                        temperature=0.7,  # 多様性のある回答を維持
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.92,
                        top_k=50,
                        repetition_penalty=1.2,  # 繰り返しを減らす
                        length_penalty=1.0,
                        no_repeat_ngram_size=3  # 大きなモデルでは3-gramの繰り返しを防止
                    )
                    
                    # 生成されたテキストをデコード
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # "回答"以降の部分を抽出
                    if "【回答】" in generated_text:
                        answer = generated_text.split("【回答】")[-1].strip()
                    elif "回答】" in generated_text:
                        answer = generated_text.split("回答】")[-1].strip()
                    elif "回答:" in generated_text:
                        answer = generated_text.split("回答:")[-1].strip()
                    elif "回答：" in generated_text:  # 全角コロンのケース
                        answer = generated_text.split("回答：")[-1].strip()
                    else:
                        # プロンプトからの続きとして回答を抽出
                        answer = generated_text[len(prompt):].strip()
                    
                    # 空の回答や短すぎる回答の場合は別の方法で生成
                    if not answer or len(answer) < 20:
                        # より単純なプロンプトで再試行
                        simple_prompt = f"""以下の情報に基づいて質問に答えてください。
情報: {relevant_context[:1500]}...

質問: {query}

回答:"""
                        
                        encoded_input = self.tokenizer(simple_prompt, return_tensors="pt", truncation=True, max_length=max_length)
                        input_ids = encoded_input["input_ids"].to(self.generation_model.device)
                        
                        output = self.generation_model.generate(
                            input_ids,
                            max_length=min(self.max_model_length, input_ids.shape[1] + 150),
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            top_k=40,
                            repetition_penalty=1.2
                        )
                        
                        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                        if "回答:" in generated_text:
                            answer = generated_text.split("回答:")[-1].strip()
                        else:
                            answer = generated_text[len(simple_prompt):].strip()
                    
                    # それでも空の場合はフォールバック
                    if not answer or len(answer) < 10:
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
                # メモリエラーなどの処理
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    print("メモリ不足エラー：より短いコンテキストで再試行します")
                    # コンテキストを短くして再試行
                    shorter_context = relevant_context[:len(relevant_context)//2]
                    
                    # 会話履歴があれば追加
                    history_prefix = ""
                    if conversation_history:
                        history_prefix = f"{conversation_history}\n"
                    
                    prompt = f"""{history_prefix}以下の情報のみを使用して質問に答えてください。

情報（一部）:
{shorter_context}

質問:
{query}

情報に含まれる内容だけを使って答えてください。情報にない内容については「情報には含まれていません」と伝えてください。

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
        context_preview = context[:500] + "..." if len(context) > 500 else context
        
        # テキスト内で一般的な情報を探す
        has_statistics_info = "統計" in context
        has_exam_info = "試験" in context or "検定" in context
        has_data_analysis = "データ分析" in context or "分析" in context

        # 様々な質問パターンに対応するフォールバック
        if "統計検定" in query:
            if "級" in query:
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
                    return f"申し訳ありませんが、提供された情報源には統計検定{level}に関する具体的な詳細が含まれていないようです。一般的には、統計検定{level}では{'確率論、多変量解析、時系列解析など高度な' if level=='1級' else '基本的な確率・統計の'}知識が問われますが、PDFの内容からは詳細を特定できませんでした。より具体的な情報は日本統計学会公式サイトで確認できます。"
            
            return f"申し訳ありませんが、提供された情報源には「{query}」に関する十分な情報が含まれていないようです。PDFには統計に関する何らかの情報が含まれているようですが、具体的な回答を見つけることができませんでした。"
                
        elif "問題" in query or "例題" in query:
            return f"申し訳ありませんが、提供された情報源には「{query}」に関する具体的な問題例や例題が見つかりませんでした。PDFには統計関連の内容が含まれているようですが、ご質問に合致する具体的な例題を特定することができませんでした。より具体的な質問内容に変更していただくか、別のPDFをアップロードしてみてください。"
        
        elif "使い方" in query or "操作" in query:
            return f"このアプリケーションは統計PDFの内容に基づいて質問に回答します。使い方は簡単です：\n1. PDFファイルを追加ボタンでアップロード\n2. 質問を入力\n3. 質問するボタンをクリック\nPDFに関連する内容であれば回答が得られます。現在のPDFには「{query}」に関する情報は含まれていないようです。"
            
        else:
            return f"申し訳ありませんが、提供された情報源からは「{query}」に対する適切な回答を見つけることができませんでした。PDFの内容を確認しましたが、この質問に直接関連する情報は含まれていないようです。別の質問をしていただくか、関連するPDFをアップロードしてみてください。"

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