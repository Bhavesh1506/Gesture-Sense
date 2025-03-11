import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            landmarks_list.extend(landmarks)
    if len(landmarks_list) > 42:
        landmarks_list = landmarks_list[:42]
    elif len(landmarks_list) < 42:
        landmarks_list.extend([[0, 0, 0]] * (42 - len(landmarks_list)))
    return np.array(landmarks_list).flatten()

gestures = {
    'hello': [], 'thank you': [], 'sorry': [], 'good morning': [], 
    'peace': [], 'thumbs up': [], 
}

gesture_history = deque(maxlen=15)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lstm_model = Sequential([
    LSTM(64, input_shape=(15, 126), return_sequences=True),
    LSTM(32),
    Dense(len(gestures), activation='softmax')
])
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


cap = cv2.VideoCapture(0)
mode = "collect"  
current_gesture = list(gestures.keys())[0]
collect_count = 0
sequence = []

print("Starting fresh data collection and training.")
print(f"Collecting data for '{current_gesture}'. Perform the gesture until 100 samples are collected.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmarks = extract_landmarks(results)
    
    if len(landmarks) > 0:
        if mode == "collect":
            gestures[current_gesture].append(landmarks)
            collect_count += 1
            
            cv2.putText(frame, f"Collecting: {current_gesture} ({collect_count}/100)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if collect_count >= 100:
                print(f"Collected 100 samples for '{current_gesture}'.")
                collect_count = 0
                current_gesture_index = list(gestures.keys()).index(current_gesture)
                current_gesture_index = (current_gesture_index + 1) % len(gestures)
                current_gesture = list(gestures.keys())[current_gesture_index]
                
                if current_gesture == list(gestures.keys())[0]:
                    mode = "train"
                    print("All gestures collected. Training the models...")
                    X = np.array([sample for gesture_samples in gestures.values() for sample in gesture_samples])
                    y = np.array([gesture for gesture, samples in gestures.items() for _ in samples])
                    
                   
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    rf_model.fit(X_train, y_train)
                    rf_pred = rf_model.predict(X_test)
                    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}")
                    
                    
                    X_lstm = np.array([X[i:i+15] for i in range(len(X)-14)])
                    y_lstm = tf.keras.utils.to_categorical([list(gestures.keys()).index(g) for g in y[14:]])
                    lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.2)
                    
                    print("Models trained. Switching to recognition mode.")
                    mode = "recognize"
                else:
                    print(f"Now collecting data for '{current_gesture}'. Perform the gesture until 100 samples are collected.")
            
        elif mode == "recognize":
            sequence.append(landmarks)
            if len(sequence) == 15:
                
                rf_pred = rf_model.predict_proba([sequence[-1]])[0]
                rf_gesture = list(gestures.keys())[np.argmax(rf_pred)]
                rf_confidence = np.max(rf_pred)
                
                
                lstm_pred = lstm_model.predict(np.array([sequence]))[0]
                lstm_gesture = list(gestures.keys())[np.argmax(lstm_pred)]
                lstm_confidence = np.max(lstm_pred)
                
                
                if rf_confidence > lstm_confidence:
                    final_gesture = rf_gesture
                    confidence = rf_confidence
                else:
                    final_gesture = lstm_gesture
                    confidence = lstm_confidence
                
                gesture_history.append(final_gesture)
                smoothed_gesture = max(set(gesture_history), key=gesture_history.count)
                
                cv2.putText(frame, f"Gesture: {smoothed_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
                if smoothed_gesture == "hello" and confidence > 0.8:
                    cv2.putText(frame, "Welcome!", (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif smoothed_gesture == "stop" and confidence > 0.8:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                elif smoothed_gesture == "zoom in" and confidence > 0.8:
                    frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
                
                sequence.pop(0)

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Advanced Hand Gesture Recognition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
