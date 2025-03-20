#!/usr/bin/env python3
"""
RAGアプリを直接起動するためのスクリプト
"""
import os
import sys

# カレントディレクトリをPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# アプリケーションをインポートして実行
if __name__ == "__main__":
    import tkinter as tk
    from src.app import RAGApp
    
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop() 