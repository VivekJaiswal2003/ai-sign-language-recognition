import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
engine.say("Voice test successful")
engine.runAndWait()
print('TTS test finished')
