import cv2
import mediapipe as mp
import json
import time

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process frame and detect hand gestures
def process_frame(frame, label, data):
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
                x = landmark.x
                y = landmark.y
                z = landmark.z
                print(f"Keypoint {idx}: ({x}, {y}, {z})")
                # Append landmark data to the label
                data[label].append({"x": x, "y": y, "z": z})
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

# Function to capture and process video
def capture_video(label):
    cap = cv2.VideoCapture(0)
    data = {label: []}
    last_update_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process frame and update data
        frame_data = process_frame(frame, label, data)
        cv2.imshow('Hand Gesture Recognition', frame_data)
        current_time = time.time()
        # Save data to JSON file if one second has passed since the last update
        if current_time - last_update_time >= 1.0:
            save_data_to_json(data)
            last_update_time = current_time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to save data to JSON file
def save_data_to_json(data):
    with open('hand_gesture_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Main function
def main():
    label = input("Enter label for the hand gesture: ")
    capture_video(label)

if __name__ == "__main__":
    main()
