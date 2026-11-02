import cv2
import mediapipe as mp
import os


# --------------------------------------------------
# Model path
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "hand_landmarker.task"
)


# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("ERROR: Hand Landmarker model not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()


print("Hand Landmarker model found:")
print(MODEL_PATH)


# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

HandLandmarkerResult = (
    mp.tasks.vision.HandLandmarkerResult
)

VisionRunningMode = mp.tasks.vision.RunningMode


# --------------------------------------------------
# Store latest detection result
# --------------------------------------------------

latest_result = None


# --------------------------------------------------
# Result callback
# --------------------------------------------------

def print_result(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    global latest_result

    latest_result = result


# --------------------------------------------------
# Hand Landmarker options
# --------------------------------------------------

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    result_callback=print_result
)


# --------------------------------------------------
# OpenCV camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open camera.")

    exit()


print()
print("Camera started.")
print("Show your hand in front of the camera.")
print("Press Q to quit.")
print()


# --------------------------------------------------
# Start MediaPipe Hand Landmarker
# --------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    timestamp_ms = 0

    while True:

        # --------------------------------------------------
        # Get camera frame
        # --------------------------------------------------

        success, frame = cap.read()


        if not success:

            print("Failed to read camera frame.")

            break


        # --------------------------------------------------
        # Mirror camera
        # --------------------------------------------------

        frame = cv2.flip(frame, 1)


        # --------------------------------------------------
        # Convert BGR -> RGB
        # --------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # --------------------------------------------------
        # Create MediaPipe image
        # --------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # --------------------------------------------------
        # Detect hand asynchronously
        # --------------------------------------------------

        timestamp_ms += 1

        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )


        # --------------------------------------------------
        # Draw detected hands
        # --------------------------------------------------

        if latest_result is not None:

            for hand_landmarks in latest_result.hand_landmarks:

                # Draw each landmark
                for landmark in hand_landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )


                # Draw connections
                connections = [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),

                    (0, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),

                    (5, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),

                    (9, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),

                    (13, 17),
                    (17, 18),
                    (18, 19),
                    (19, 20),

                    (0, 17)
                ]


                for start, end in connections:

                    x1 = int(
                        hand_landmarks[start].x
                        * frame.shape[1]
                    )

                    y1 = int(
                        hand_landmarks[start].y
                        * frame.shape[0]
                    )

                    x2 = int(
                        hand_landmarks[end].x
                        * frame.shape[1]
                    )

                    y2 = int(
                        hand_landmarks[end].y
                        * frame.shape[0]
                    )

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


        # --------------------------------------------------
        # Display number of detected hands
        # --------------------------------------------------

        if latest_result is not None:

            hand_count = len(
                latest_result.hand_landmarks
            )

            cv2.putText(
                frame,
                f"Hands: {hand_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        # --------------------------------------------------
        # Display camera
        # --------------------------------------------------

        cv2.imshow(
            "MediaPipe Hand Detection",
            frame
        )


        # --------------------------------------------------
        # Q = quit
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

print("Hand detection stopped.")
