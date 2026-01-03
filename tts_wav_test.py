import pyttsx3
import os
import time

OUT = 'tts_test_output.wav'
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    print('Saving WAV to', OUT)
    engine.save_to_file('This is a WAV test from pyttsx3.', OUT)
    engine.runAndWait()
    print('Saved. Attempting to play via winsound...')
    import winsound
    winsound.PlaySound(OUT, winsound.SND_FILENAME)
    print('Play finished.')
except Exception as e:
    print('Error during WAV test:', e)
finally:
    try:
        if os.path.exists(OUT):
            os.remove(OUT)
    except Exception:
        pass
