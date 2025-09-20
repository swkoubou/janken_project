import requests
import json
import time

# ロボットのIPアドレスとポート番号
# ネットワークに接続したPepperのIPアドレスに置き換えてください
# ネットワークに繋がってたらブラウザでIPアドレスを打ち込むとPepperのWebページに繋がる
PEPPER_IP = "10.94.15.162" 
PEPPER_PORT = "80"

# naoqiのREST APIのベースURL
BASE_URL = f"http://{PEPPER_IP}:{PEPPER_PORT}/api"

def say_text(text_to_say):
    """
    Pepperに指定されたテキストを話させる関数
    """
    try:
        # TTS (Text-to-Speech) サービスのAPIエンドポイント
        url = f"{BASE_URL}/tts"
        
        # リクエストボディ（JSON形式）
        data = {
            "text": text_to_say,
            "speed": 100,  # 話す速度
            "volume": 0.8  # 音量
        }
        
        # POSTリクエストを送信
        print(f"Pepperに「{text_to_say}」と言わせます...")
        response = requests.post(url, json=data, timeout=5)
        
        # レスポンスのステータスコードを確認
        response.raise_for_status()
        
        print("コマンドが正常に送信されました。")
        
    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")
        print("PepperのIPアドレスが正しいか、ネットワークに接続されているか確認してください。")

if __name__ == "__main__":
    # テキストを話させる
    say_text("こんにちは、私はペッパーです。")
    time.sleep(3)  # 3秒待機
    
    say_text("Web経由で私を動かしています。")
    time.sleep(3)

    say_text("これは便利な機能ですね。")