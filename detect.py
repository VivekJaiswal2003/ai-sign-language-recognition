import argparse
import threading
from collections import deque, Counter
import time

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3


# -------------------------------------------------------
# NORMALIZATION (same as train.py + mark.py)
# -------------------------------------------------------
def normalize_landmarks_single(landmarks):
    try:
        arr = np.array(landmarks, dtype=float).reshape(-1, 3)
        wrist = arr[0].copy()

        arr[:, 0:2] -= wrist[0:2]
        arr[:, 2] -= wrist[2]

        dists = np.linalg.norm(arr[:, 0:2], axis=1)
        scale = dists.max() if np.isfinite(dists.max()) and dists.max() > 1e-6 else 1.0

        arr[:, 0:2] /= scale
        arr[:, 2] /= scale

        return arr.flatten().tolist()

    except Exception:
        return landmarks


# -------------------------------------------------------
# TEXT TO SPEECH (async)
# -------------------------------------------------------
def speak_async(text, lock, rate=120, volume=0.9):
    def _worker():
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except:
                pass

            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            with lock:
                engine.say(text)
                engine.runAndWait()

        except:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# -------------------------------------------------------
# MAIN DETECTION LOOP
# -------------------------------------------------------
def main(camera_index=0,
         model_path="asl_sign_model.pkl",
         scaler_path="asl_scaler.pkl",
         encoder_path="asl_label_encoder.pkl",
         stability_frames=30,
         tts_rate=120,
         tts_volume=0.9,
         speak_cooldown=3.0):

    # Load trained model + scaler + label encoder
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)   # 🔥 IMPORTANT (fixes wrong predictions)

    tts_lock = threading.Lock()

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("❌ Cannot open webcam.")

    cv2.namedWindow("ASL Detection", cv2.WINDOW_NORMAL)

    recent_preds = deque(maxlen=stability_frames)
    last_spoken = None
    last_spoken_time = 0

    print("🚀 Detection started... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read camera frame.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            predicted_label = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                if len(landmarks) == 63:
                    normalized = normalize_landmarks_single(landmarks)
                    X = scaler.transform([normalized])

                    pred_enc = model.predict(X)[0]
                    predicted_label = le.inverse_transform([pred_enc])[0]   # 🔥 FIXED

                    cv2.putText(frame, f"Sign: {predicted_label}",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 2)

            # Stabilizer logic
            if predicted_label is None:
                recent_preds.clear()
            else:
                recent_preds.append(predicted_label)

                if len(recent_preds) == recent_preds.maxlen:
                    top, count = Counter(recent_preds).most_common(1)[0]

                    if count >= (stability_frames // 2 + 1):
                        now = time.time()

                        if (top != last_spoken) or (now - last_spoken_time > speak_cooldown):
                            speak_async(top, tts_lock, tts_rate, tts_volume)
                            last_spoken = top
                            last_spoken_time = now

            cv2.imshow("ASL Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=str, default="asl_sign_model.pkl")
    parser.add_argument("--scaler", type=str, default="asl_scaler.pkl")
    parser.add_argument("--encoder", type=str, default="asl_label_encoder.pkl")
    parser.add_argument("--stability", type=int, default=30)
    args = parser.parse_args()

    main(
        camera_index=args.camera,
        model_path=args.model,
        scaler_path=args.scaler,
        encoder_path=args.encoder,
        stability_frames=args.stability
    )
