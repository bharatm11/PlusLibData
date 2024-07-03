# This script reads from the epiphan frame grabber directly and displays the frame.


import cv2

def main():
    # Initialize the capture device
    capture = cv2.VideoCapture(1)  # Use the correct index for your Epiphan VGA2USB device

    if not capture.isOpened():
        print("Failed to open capture device")
        return

    # Set the resolution and frame rate (if supported by the device)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width, adjust as needed
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height, adjust as needed
    capture.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate, adjust as needed

    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Edit the frame (e.g., convert to grayscale)
        edited_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the edited frame
        cv2.imshow('Edited Frame', edited_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture device and close any open windows
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
