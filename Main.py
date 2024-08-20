import ctypes
import cv2
from cvzone.HandTrackingModule import HandDetector
import socket
import logging

# Constants
ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 1280, 720
SCALE_FACTOR = 0.5
OFFSET = 85
RIGHT_HAND_PORT = 5052
LEFT_HAND_PORT = 5053
SCREEN_UPDATE_RATE = 1  # in milliseconds

# Configure logging
logging.basicConfig(level=logging.INFO)

def hide_console():
    """Hide the console window."""
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)  # 0 = SW_HIDE

def get_screen_dimensions():
    """Get the screen width and height."""
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    return screen_width, screen_height

def initialize_camera(width, height):
    """Initialize the camera with the specified width and height."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def calculate_window_position(screen_width, screen_height, width, height, offset):
    """Calculate the position of the window on the screen."""
    window_x = (screen_width - width) // 2
    window_y = screen_height - height - offset
    return window_x, window_y

def create_default_hand_data():
    """Create default hand data with all landmarks set to (0, 0, 0)."""
    return [(0, 0, 0)] * 21  # 21 landmarks with (x, y, z) = (0, 0, 0)

def format_hand_data(landmarks, original_height):
    """Format the hand landmarks into a list of (x, y, z) tuples."""
    return [(lm[0], original_height - lm[1], lm[2]) for lm in landmarks]

def detect_and_format_hand_data(detector, img, original_height):
    """
    Detect hands in the image and format their data.
    
    Returns the processed image and a dictionary with hand data for each port.
    """
    hands, img = detector.findHands(img)

    # Default hand data
    default_hand_data = create_default_hand_data()
    data_to_send = {
        RIGHT_HAND_PORT: default_hand_data.copy(),
        LEFT_HAND_PORT: default_hand_data.copy()
    }

    if hands:
        for hand in hands:
            hand_type = hand['type']
            landmarks = hand['lmList']
            hand_data = format_hand_data(landmarks, original_height)

            port = RIGHT_HAND_PORT if hand_type == 'Right' else LEFT_HAND_PORT
            data_to_send[port] = hand_data

    return img, data_to_send

def send_hand_data(sock, data_to_send):
    """Send the formatted hand data to the corresponding ports."""
    for port, data in data_to_send.items():
        try:
            sock.sendto(str.encode(str(data)), ("127.0.0.1", port))
        except Exception as e:
            logging.error(f"Failed to send data on port {port}: {e}")

def initialize_components():
    """Initialize all necessary components for the application."""
    hide_console()
    
    width = int(ORIGINAL_WIDTH * SCALE_FACTOR)
    height = int(ORIGINAL_HEIGHT * SCALE_FACTOR)
    
    cap = initialize_camera(ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
    detector = HandDetector(maxHands=2, detectionCon=0.8)
    
    screen_width, screen_height = get_screen_dimensions()
    window_x, window_y = calculate_window_position(screen_width, screen_height, width, height, OFFSET)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", width, height)
    cv2.moveWindow("Image", window_x, window_y)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    return cap, detector, sock, width, height

def main_loop(cap, detector, sock, width, height):
    """Main loop to process video frames and send hand data."""
    while True:
        success, img = cap.read()
        if not success:
            logging.error("Failed to grab frame")
            break

        # Flip the image horizontally
        img = cv2.flip(img, 1)

        # Detect hands and format the data
        img, data_to_send = detect_and_format_hand_data(detector, img, ORIGINAL_HEIGHT)

        # Send data to corresponding ports
        send_hand_data(sock, data_to_send)

        # Resize and display the image
        img = cv2.resize(img, (width, height))
        cv2.imshow("Image", img)

        if cv2.waitKey(SCREEN_UPDATE_RATE) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to initialize components and start the main loop."""
    cap, detector, sock, width, height = initialize_components()
    main_loop(cap, detector, sock, width, height)

if __name__ == "__main__":
    main()