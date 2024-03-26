import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to collect labeled hand gesture data
def collect_data(label, num_samples):
    cap = cv2.VideoCapture(0)
    sample_count = 0

    while cap.isOpened() and sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB and process it using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('Collecting Data - Press "Q" to Quit', frame)

        # Save the frame with the corresponding label
        if cv2.waitKey(1) & 0xFF == ord('s'):
            filename = f'data/{label}_{sample_count}.jpg'  # Save the image with a unique filename
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')
            sample_count += 1

        # Quit if 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Specify the label and number of samples to collect for each gesture
    label = 'thumbs_up'
    num_samples = 100

    # Collect data for the specified gesture
    collect_data(label, num_samples)

if __name__ == "__main__":
    main()
