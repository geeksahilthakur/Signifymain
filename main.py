import cv2
import mediapipe as mp

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=4, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define custom gestures and their corresponding values
CUSTOM_GESTURES = {
    'Fist': 'hello',
    'One': 'i',
    'Two': 'am',
    'Three': 'good',
    'Four': 'bad',
    'Five': 'yes',
}


# Function to process frame and detect hand gestures
def process_frame(frame):
    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image using Mediapipe
    results = hands.process(frame_rgb)
    # Check if hand(s) detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract gesture
            landmark_list = []
            for landmark in hand_landmarks.landmark:
                landmark_list.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                })
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Recognize gesture
            # Check if hand classification results are available
            if results.multi_handedness and results.multi_handedness[0].classification:
                label_index = results.multi_handedness[0].classification[0].index
                if label_index in CUSTOM_GESTURES:
                    gesture_label = CUSTOM_GESTURES[label_index]
                    cv2.putText(frame, gesture_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'Data To be Trained.....', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
    return frame


# Main loop for capturing and processing video
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    cv2.imshow('Hand Gesture Recognition', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
