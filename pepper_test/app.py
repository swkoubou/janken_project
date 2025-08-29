from naoqi import ALProxy

# Pepper の IP とポートを指定
PEPPER_IP = "192.168.xxx.xxx"
PORT = 9559

tts = ALProxy("ALTextToSpeech", PEPPER_IP, PORT)
tts.say("こんにちは、Python から接続できました")
