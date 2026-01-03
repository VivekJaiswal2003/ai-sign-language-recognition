import pyttsx3
import threading


def list_and_sync():
    try:
        engine = pyttsx3.init()
        print("Engine created. Voices:")
        for v in engine.getProperty('voices'):
            print(" -", v.id, "|", v.name)
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        print("Synchronous speech: 'Synchronous test now'")
        engine.say("Synchronous test now")
        engine.runAndWait()
        print("Synchronous done.")
    except Exception as e:
        print("Error creating synchronous engine:", e)


def threaded_test(text):
    try:
        import pythoncom
        pythoncom.CoInitialize()
        eng = pyttsx3.init()
        eng.setProperty('rate', 150)
        eng.setProperty('volume', 0.9)
        print("🔊 Thread worker speaking:", text)
        eng.say(text)
        eng.runAndWait()
    except Exception as e:
        print("TTS worker error:", e)
    finally:
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass


if __name__ == '__main__':
    list_and_sync()
    t = threading.Thread(target=threaded_test, args=("Threaded test now",), daemon=True)
    t.start()
    t.join(timeout=10)
    print("Thread test done.")
