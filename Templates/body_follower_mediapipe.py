import cv2
import mediapipe as mp
import os


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "pose_landmarker.task"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Pose Landmarker model not found!")
    print()
    print("Expected location:")
    print(MODEL_PATH)

    raise SystemExit


print("Pose Landmarker model found:")
print(MODEL_PATH)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

PoseLandmarkerResult = (
    mp.tasks.vision.PoseLandmarkerResult
)

VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# STORE LATEST DETECTION RESULT
# ============================================================

latest_result = None


# ============================================================
# RESULT CALLBACK
# ============================================================

def print_result(
    result: PoseLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):

    global latest_result

    latest_result = result


# ============================================================
# POSE LANDMARKER OPTIONS
# ============================================================

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    # Detect one person
    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    result_callback=print_result
)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open camera.")

    raise SystemExit


print()
print("Camera started.")
print("Show your full body in front of the camera.")
print("Press Q to quit.")
print()


# ============================================================
# POSE CONNECTIONS
# ============================================================
#
# MediaPipe Pose has 33 landmarks.
#
# These connections create the body skeleton.
#

POSE_CONNECTIONS = [

    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Shoulders
    (9, 10),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # Left hand/fingers
    (15, 17),
    (15, 19),
    (15, 21),

    # Right hand/fingers
    (16, 18),
    (16, 20),
    (16, 22),

    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),

    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),

    # Feet
    (27, 29),
    (28, 30)
]


# ============================================================
# START POSE LANDMARKER
# ============================================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    timestamp_ms = 0


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:


        # ====================================================
        # GET CAMERA FRAME
        # ====================================================

        success, frame = cap.read()


        if not success:

            print(
                "Failed to read camera frame."
            )

            break


        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(frame, 1)


        # ====================================================
        # FRAME SIZE
        # ====================================================

        height, width, _ = frame.shape


        # ====================================================
        # CONVERT BGR -> RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # CREATE MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # DETECT POSE ASYNCHRONOUSLY
        # ====================================================

        timestamp_ms += 1

        landmarker.detect_async(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # DRAW DETECTED BODY
        # ====================================================

        if latest_result is not None:

            # ------------------------------------------------
            # Number of detected people
            # ------------------------------------------------

            pose_count = len(
                latest_result.pose_landmarks
            )


            # ------------------------------------------------
            # Draw each detected pose
            # ------------------------------------------------

            for pose_landmarks in (
                latest_result.pose_landmarks
            ):


                # ============================================
                # DRAW LANDMARKS
                # ============================================

                for landmark in pose_landmarks:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )


                    # Only draw landmarks that are
                    # inside the camera image.

                    if (
                        0 <= x < width
                        and
                        0 <= y < height
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 255, 0),
                            -1
                        )


                # ============================================
                # DRAW BODY CONNECTIONS
                # ============================================

                for start, end in POSE_CONNECTIONS:

                    if (
                        start >= len(pose_landmarks)
                        or
                        end >= len(pose_landmarks)
                    ):

                        continue


                    x1 = int(
                        pose_landmarks[start].x
                        * width
                    )

                    y1 = int(
                        pose_landmarks[start].y
                        * height
                    )


                    x2 = int(
                        pose_landmarks[end].x
                        * width
                    )

                    y2 = int(
                        pose_landmarks[end].y
                        * height
                    )


                    # Draw connection only if
                    # coordinates are valid.

                    if (
                        0 <= x1 < width
                        and
                        0 <= y1 < height
                        and
                        0 <= x2 < width
                        and
                        0 <= y2 < height
                    ):

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )


                # ============================================
                # CALCULATE BODY CENTER
                # ============================================
                #
                # Use shoulders + hips to estimate
                # the person's body center.
                #
                # 11 = left shoulder
                # 12 = right shoulder
                # 23 = left hip
                # 24 = right hip

                if len(pose_landmarks) >= 25:

                    body_center_x = (
                        pose_landmarks[11].x
                        +
                        pose_landmarks[12].x
                        +
                        pose_landmarks[23].x
                        +
                        pose_landmarks[24].x
                    ) / 4.0


                    body_center_y = (
                        pose_landmarks[11].y
                        +
                        pose_landmarks[12].y
                        +
                        pose_landmarks[23].y
                        +
                        pose_landmarks[24].y
                    ) / 4.0


                    center_pixel_x = int(
                        body_center_x * width
                    )

                    center_pixel_y = int(
                        body_center_y * height
                    )


                    # ========================================
                    # DRAW BODY CENTER
                    # ========================================

                    cv2.circle(
                        frame,
                        (
                            center_pixel_x,
                            center_pixel_y
                        ),
                        12,
                        (0, 0, 255),
                        -1
                    )


                    # ========================================
                    # DISPLAY BODY CENTER
                    # ========================================

                    cv2.putText(
                        frame,
                        "BODY CENTER",
                        (
                            center_pixel_x + 15,
                            center_pixel_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )


                    # ========================================
                    # DISPLAY X POSITION
                    # ========================================

                    cv2.putText(
                        frame,
                        f"Body X: {body_center_x:.2f}",
                        (
                            20,
                            height - 70
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )


                    # ========================================
                    # BODY POSITION
                    # ========================================

                    if body_center_x < 0.40:

                        body_position = "LEFT"


                    elif body_center_x > 0.60:

                        body_position = "RIGHT"


                    else:

                        body_position = "CENTER"


                    # ========================================
                    # DISPLAY POSITION
                    # ========================================

                    cv2.putText(
                        frame,
                        f"POSITION: {body_position}",
                        (
                            20,
                            height - 30
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )


            # =================================================
            # DISPLAY NUMBER OF PEOPLE
            # =================================================

            cv2.putText(
                frame,
                f"People: {pose_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        else:

            # =================================================
            # NO POSE
            # =================================================

            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (
                    20,
                    40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # ====================================================
        # DISPLAY CAMERA
        # ====================================================

        cv2.imshow(
            "MediaPipe Body Detection",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Body detection stopped.")
