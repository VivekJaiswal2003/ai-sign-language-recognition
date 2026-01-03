import argparse
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time


# -------------------------------------------------------
# SAME NORMALIZATION USED IN TRAIN.PY AND DETECT.PY
# -------------------------------------------------------
def normalize_landmarks_single(landmarks):
    """Normalize a single list of 63 raw mediapipe hand landmarks"""
    arr = np.array(landmarks, dtype=float).reshape(-1, 3)

    wrist = arr[0].copy()

    # Shift wrist to origin (x,y,z)
    arr[:, 0] -= wrist[0]
    arr[:, 1] -= wrist[1]
    arr[:, 2] -= wrist[2]

    # Scaling based on max XY distance from wrist
    dists = np.linalg.norm(arr[:, :2], axis=1)
    max_dist = dists.max()

    if not np.isfinite(max_dist) or max_dist <= 1e-6:
        max_dist = 1.0

    arr[:, 0] /= max_dist
    arr[:, 1] /= max_dist
    arr[:, 2] /= max_dist

    return arr.flatten()


# -------------------------------------------------------
# LANDMARK COLLECTOR
# -------------------------------------------------------
class LandmarkDataCollector:
    def __init__(self, camera_id: int = 0):

        # MediaPipe Hands init
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Your class labels
        self.classes = [
            'HELLO', 'GOOD MORNING TO ALL', 'HAVE A NICE DAY','RESCUE',
            'HELP', 'GOOD NIGHT', 'I AM VIVEK KUMAR', 'I AM VINAY KUMAR','I AM TUSHAR SHARMA','I AM VINIT',
            'I AM VISHAL', 'NEED PEN', 'HOW ARE YOU SIR', 'FIVE GROUP MEMBERS', 'I AM HUNGURY',
        ]

        self.current_class = self.classes[0]
        self.samples_collected = 0
        self.target_samples = 300
        self.camera_id = camera_id

        # Create CSV dataset file
        self.create_dataset_file()

    # -------------------------------------------------------
    def create_dataset_file(self):
        if not os.path.exists('asl_landmark_dataset.csv'):
            header = []
            for i in range(21):
                header.extend([
                    f'landmark_{i}_x',
                    f'landmark_{i}_y',
                    f'landmark_{i}_z'
                ])
            header.append("class_label")

            df = pd.DataFrame(columns=header)
            df.to_csv("asl_landmark_dataset.csv", index=False)
            print("✅ Created new dataset file asl_landmark_dataset.csv")

    # -------------------------------------------------------
    def extract_landmark_vector(self, hand_landmarks):

        raw = []
        for lm in hand_landmarks.landmark:
            raw.extend([lm.x, lm.y, lm.z])

        normalized = normalize_landmarks_single(raw)
        return np.array(normalized)

    # -------------------------------------------------------
    def save_landmark_data(self, landmark_vector, class_label):

        row_data = list(landmark_vector) + [class_label]

        header = []
        for i in range(21):
            header.extend([
                f'landmark_{i}_x',
                f'landmark_{i}_y',
                f'landmark_{i}_z'
            ])
        header.append("class_label")

        df = pd.DataFrame([row_data], columns=header)
        df.to_csv("asl_landmark_dataset.csv", mode='a', header=False, index=False)

        self.samples_collected += 1
        return True

    # -------------------------------------------------------
    def draw_landmarks_and_info(self, frame, hand_landmarks, collecting):

        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

        h, w = frame.shape[:2]
        x_vals = [lm.x * w for lm in hand_landmarks.landmark]
        y_vals = [lm.y * h for lm in hand_landmarks.landmark]

        x1, x2 = int(min(x_vals)), int(max(x_vals))
        y1, y2 = int(min(y_vals)), int(max(y_vals))

        cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)

        cv2.putText(frame, f"Class: {self.current_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"Samples: {self.samples_collected}/{self.target_samples}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        status = "Collecting" if collecting else "Paused"
        cv2.putText(frame, f"Status: {status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return frame

    # -------------------------------------------------------
    def run_collection(self):

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("❌ Camera not found!")
            return

        auto_collect = False
        last_save_time = 0
        save_interval = 0.5

        print("🚀 Starting Data Collection")
        print("SPACE = Start/Stop   | S = Single Save | N = Next | P = Prev | Q = Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            collecting = auto_collect

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                frame = self.draw_landmarks_and_info(frame, lm, collecting)

                vec = self.extract_landmark_vector(lm)

                if auto_collect:
                    cur_time = time.time()
                    if cur_time - last_save_time >= save_interval:
                        self.save_landmark_data(vec, self.current_class)
                        last_save_time = cur_time
                        print(f"Saved {self.current_class} → {self.samples_collected}")

                        if self.samples_collected >= self.target_samples:
                            print(f"🎉 Completed {self.current_class}")
                            self.move_to_next_class()
                            auto_collect = False
            else:
                cv2.putText(frame, "Hand: Not Detected", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("ASL Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                auto_collect = not auto_collect
            elif key == ord('s') and result.multi_hand_landmarks:
                self.save_landmark_data(vec, self.current_class)
                print("📸 Single sample saved.")
            elif key == ord('n'):
                self.move_to_next_class()
            elif key == ord('p'):
                self.move_to_previous_class()

        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------------------------------
    def move_to_next_class(self):
        idx = self.classes.index(self.current_class)
        if idx + 1 < len(self.classes):
            self.current_class = self.classes[idx + 1]
            self.samples_collected = 0
        else:
            print("🎉 ALL CLASSES DONE!")

    def move_to_previous_class(self):
        idx = self.classes.index(self.current_class)
        if idx > 0:
            self.current_class = self.classes[idx - 1]
            self.samples_collected = 0


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0)
    args = parser.parse_args()

    ld = LandmarkDataCollector(camera_id=args.camera)
    ld.run_collection()
