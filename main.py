import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f'変更が検出されました: {event.src_path}')
            os.execv(sys.executable, ['python'] + sys.argv)

if __name__ == "__main__":
    path = '.'  # 現在のディレクトリを監視
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    # アプリを起動
    subprocess.Popen(['python', 'src/app.py'])

    try:
        while True:
            time.sleep(1)  # メインスレッドを維持
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
