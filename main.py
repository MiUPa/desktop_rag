import os
os.environ["TRANSFORMERS_NO_SAFE_TENSORS"] = "1"

# その後で他のモジュールをインポートする
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # カレントディレクトリをパスに追加

from src.app import RAGApp
import time
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, process):
        self.process = process

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f'変更が検出されました: {event.src_path}')
            if self.process:
                print("既存のアプリを終了します...")
                self.process.terminate()  # 既存のプロセスを終了
                self.process.wait()       # プロセスが終了するのを待つ
            print("新しいアプリを起動します...")
            os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    path = '.'  # 現在のディレクトリを監視
    # 環境変数を設定してPYTHONPATHにカレントディレクトリを追加
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # 環境変数を設定してプロセスを起動
    process = subprocess.Popen(['python', 'src/app.py'], env=env)
    
    event_handler = ChangeHandler(process)
    
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)  # メインスレッドを維持
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
