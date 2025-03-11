import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks
def extract_landmarks(results):
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None

# Initialize gestures and classifier
gestures = {
    'hello': [],
    'thank you': [],
    'sorry': [],
    'good morning': [],
    'call me': []
}
knn = KNeighborsClassifier(n_neighbors=3)

# Main loop for data collection and recognition
cap = cv2.VideoCapture(0)
mode = "collect"  # Start in data collection mode
current_gesture = list(gestures.keys())[0]
collect_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        landmarks = extract_landmarks(results)
        if landmarks is not None:
            if mode == "collect":
                gestures[current_gesture].append(landmarks)
                collect_count += 1
                if collect_count >= 100:
                    collect_count = 0
                    current_gesture_index = list(gestures.keys()).index(current_gesture)
                    current_gesture_index = (current_gesture_index + 1) % len(gestures)
                    current_gesture = list(gestures.keys())[current_gesture_index]
                    
                    if current_gesture == list(gestures.keys())[0]:
                        mode = "train"
                        print("Training the model...")
                        X = []
                        y = []
                        for gesture, samples in gestures.items():
                            X.extend(samples)
                            y.extend([gesture] * len(samples))
                        knn.fit(X, y)
                        print("Model trained. Switching to recognition mode.")
                        mode = "recognize"
            elif mode == "recognize":
                gesture = knn.predict([landmarks])[0]
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if mode == "collect":
        cv2.putText(frame, f"Collecting: {current_gesture} ({collect_count}/100)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif mode == "train":
        cv2.putText(frame, "Training model...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        mode = "collect"
        gestures = {gesture: [] for gesture in gestures}
        current_gesture = list(gestures.keys())[0]
        collect_count = 0
        print("Reset. Starting data collection again.")

cap.release()
cv2.destroyAllWindows()