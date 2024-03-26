import cv2
import mediapipe as mp
import json
import numpy as np
import pyttsx3

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1)  # Lowering the confidence threshold
mp_drawing = mp.solutions.drawing_utils

# Function to read hand gesture data from JSON file
def read_hand_gesture_data(filename):
    with open(filename, 'r') as file:
        hand_gesture_data = json.load(file)
    return hand_gesture_data

# Function to compare hand landmarks with hand gesture data
def compare_hand_landmarks(hand_landmarks, hand_gesture_data):
    # Extract landmark coordinates
    hand_landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])

    for label, gestures in hand_gesture_data.items():
        for gesture_data in gestures:
            # Ensure gesture data has the correct length
            if len(gesture_data) != len(hand_landmarks_np):
                continue  # Skip if the length is incorrect

            # Convert gesture data to numpy array
            gesture_data_np = np.array(gesture_data)

            # Calculate Euclidean distances between hand landmarks and gesture data
            distances = np.linalg.norm(hand_landmarks_np - gesture_data_np, axis=1)
            avg_distance = np.mean(distances)

            # Set a threshold for similarity
            if avg_distance < 0.1:  # Adjust the threshold as needed
                return label
    return None

# Function to process frame and detect hand gestures
def process_frame(frame, hand_gesture_data):
    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image using Mediapipe
    results = hands.process(frame_rgb)
    # Check if hand(s) detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Print detected keypoints
            print("Detected keypoints:")
            for idx, landmark in enumerate(hand_landmarks.landmark):
                print(f"Keypoint {idx}: ({landmark.x}, {landmark.y}, {landmark.z})")
            # Compare hand landmarks with hand gesture data
            label = compare_hand_landmarks(hand_landmarks.landmark, hand_gesture_data)
            if label:
                print("Detected hand gesture:", label)
                speak_label(label)
                return label  # Return the label if a gesture matches
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return None

# Function to speak the detected label
def speak_label(label):
    engine.say(label)
    engine.runAndWait()

# Main loop for capturing and processing video
def main():
    hand_gesture_data = read_hand_gesture_data('hand_gesture_data.json')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label = process_frame(frame, hand_gesture_data)
        if label:
            print("Detected hand gesture:", label)  # Print the detected label
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
