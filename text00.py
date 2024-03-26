import cv2
import mediapipe as mp

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to process frame and detect hand gestures
def process_frame(frame):
    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image using Mediapipe
    results = hands.process(frame_rgb)
    # Check if hand(s) detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract and print landmark coordinates
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                print(f"Landmark - X: {x}, Y: {y}, Z: {z}")
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return frame

# Main loop for capturing and processing video
def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow('Hand Gesture Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
