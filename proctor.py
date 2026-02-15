import cv2
import mediapipe as mp
import numpy as np
import time

# ---------------- CONFIG ---------------- #
YAW_THRESHOLD = 12
PITCH_THRESHOLD = 12
EVENT_COOLDOWN = 1.5
HOLD_TIME = 1.0  # seconds to confirm distraction
# ---------------------------------------- #

class ProctoringSystem:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.suspicion_score = 0
        self.last_event_time = 0
        self.prev_yaw = 0
        self.prev_pitch = 0
        self.distraction_start = None

    def increment_score(self):
        if time.time() - self.last_event_time > EVENT_COOLDOWN:
            self.suspicion_score += 1
            self.last_event_time = time.time()

    def draw_label(self, frame, text, position, bg_color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = position
        cv2.rectangle(frame, (x-5, y-h-10), (x+w+5, y+5), bg_color, -1)
        cv2.putText(frame, text, (x, y),
                    font, scale, (255,255,255), thickness)

    def process_frame(self, frame):

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        status = "Forward"
        color = (0, 150, 0)

        if results.multi_face_landmarks:

            if len(results.multi_face_landmarks) > 1:
                status = "MULTIPLE FACES"
                color = (0, 0, 255)
                self.increment_score()

            for face_landmarks in results.multi_face_landmarks[:1]:

                h, w, _ = frame.shape
                nose = face_landmarks.landmark[1]

                x = int(nose.x * w)
                y = int(nose.y * h)

                center_x = w // 2
                center_y = h // 2

                yaw = (x - center_x) / 10
                pitch = (center_y - y) / 10

                # ---------- SMOOTHING ----------
                yaw = 0.7 * self.prev_yaw + 0.3 * yaw
                pitch = 0.7 * self.prev_pitch + 0.3 * pitch
                self.prev_yaw = yaw
                self.prev_pitch = pitch

                distracted = False

                # ---------- MIRROR FIX ----------
                if yaw > YAW_THRESHOLD:
                    status = "Looking RIGHT"
                    distracted = True
                elif yaw < -YAW_THRESHOLD:
                    status = "Looking LEFT"
                    distracted = True
                elif pitch > PITCH_THRESHOLD:
                    status = "Looking DOWN"
                    distracted = True
                else:
                    status = "Forward"

                # ---------- HOLD CONFIRMATION ----------
                if distracted:
                    color = (0, 0, 255)
                    if self.distraction_start is None:
                        self.distraction_start = time.time()
                    elif time.time() - self.distraction_start > HOLD_TIME:
                        self.increment_score()
                else:
                    self.distraction_start = None

                # ---------- DRAW DIRECTION LINE ----------
                length = 100
                x2 = int(x + length * np.sin(np.radians(yaw)))
                y2 = int(y - length * np.sin(np.radians(pitch)))
                cv2.line(frame, (x, y), (x2, y2), (255, 0, 0), 3)

                # Display angles
                self.draw_label(frame,
                                f"Yaw:{int(yaw)} Pitch:{int(pitch)}",
                                (20, 40),
                                (50, 50, 50))

        else:
            status = "NO FACE"
            color = (0, 0, 255)
            self.increment_score()

        # Display status & score
        self.draw_label(frame, status, (20, 80), color)
        self.draw_label(frame,
                        f"Suspicion Score: {self.suspicion_score}",
                        (20, 120),
                        (0, 120, 255))

        return frame


def main():
    cap = cv2.VideoCapture(0)
    system = ProctoringSystem()

    print("AI Proctoring System Running... Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = system.process_frame(frame)
        cv2.imshow("AI Proctoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
