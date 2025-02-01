from sentence_transformers import SentenceTransformer, util
import torch

# SentenceTransformerモデルの読み込み（例: 'all-MiniLM-L6-v2'）
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def rank_results(query, texts, top_k=3):
    """
    PDFテキストのリスト(texts)から、問い合わせに関連する上位の段落を返す。
    textsは各段落のリストとする。
    """
    # 質問と各段落の埋め込みを取得
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    text_embeddings = embedder.encode(texts, convert_to_tensor=True)
    
    # コサイン類似度でランキング
    cosine_scores = util.cos_sim(query_embedding, text_embeddings)[0]
    
    # 上位top_kのインデックス取得
    top_results = torch.topk(cosine_scores, k=top_k)
    top_indices = top_results.indices.cpu().numpy().tolist()
    
    # 上位の段落を結合して返す
    ranked_text = " ".join([texts[idx] for idx in top_indices])
    return ranked_text
