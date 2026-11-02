import cv2
import mediapipe as mp
import os
import time

from pysimverse import Drone


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "hand_landmarker.task"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Hand Landmarker model not found!")
    print("Expected location:")
    print(MODEL_PATH)
    raise SystemExit


print("Hand Landmarker model found:")
print(MODEL_PATH)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

HandLandmarkerResult = (
    mp.tasks.vision.HandLandmarkerResult
)

VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# LATEST HAND DATA
# ============================================================

latest_result = None
latest_result_time = 0.0


# ============================================================
# MEDIAPIPE CALLBACK
# ============================================================

def hand_result_callback(
    result: HandLandmarkerResult,
    output_image: mp.Image,
    timestamp_ms: int
):

    global latest_result
    global latest_result_time

    latest_result = result
    latest_result_time = time.monotonic()


# ============================================================
# MEDIAPIPE OPTIONS
# ============================================================

options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    result_callback=hand_result_callback
)


# ============================================================
# DRONE CONTROL SETTINGS
# ============================================================

# IMPORTANT:
#
# Use normal RC speed values.
#
# Start with:
#
# 30 = slow
# 40 = medium
# 50 = faster
#
# Do NOT start with 500.

MAX_SIDE_SPEED = 50


# Minimum speed after leaving the center zone.
#
# This makes the drone respond immediately instead of
# producing tiny values such as 2, 5, 8.

MIN_SIDE_SPEED = 20


# ============================================================
# CENTER DANGER / STOP ZONE
# ============================================================
#
# Center 10% of screen is STOP.
#
# 0.45 -> 0.55
#
# Hand here:
#
#       STOP
#
# Outside:
#
# LEFT       RIGHT

CENTER_DEADZONE = 0.10


# ============================================================
# SPEED RESPONSE
# ============================================================
#
# Hand farther from center = faster movement.
#
# At 0.45 distance from center -> maximum speed.

MAX_HAND_DISTANCE = 0.45


# ============================================================
# OTHER RC AXES
# ============================================================

FORWARD_BACKWARD = 0
UP_DOWN = 0
YAW = 0


# ============================================================
# LEFT / RIGHT DIRECTION
# ============================================================
#
# Standard expected mapping:
#
# negative = LEFT
# positive = RIGHT
#
# If your drone moves opposite to the displayed direction,
# simply swap these two values.

LEFT_DIRECTION = -1
RIGHT_DIRECTION = 1


# ============================================================
# HAND POSITION FILTER
# ============================================================
#
# Smaller value:
#     more responsive
#
# Larger value:
#     smoother but slower
#
# 0.35 is a good compromise.

SMOOTHING = 0.35


# ============================================================
# HAND RESULT TIMEOUT
# ============================================================
#
# If MediaPipe stops providing results for this amount of
# time, immediately stop the drone.
#
# This prevents the drone from continuing with an old command.

RESULT_TIMEOUT = 0.20


# ============================================================
# CONNECT DRONE
# ============================================================

print()
print("Connecting to drone...")

drone = Drone()

drone.connect()

print("Drone connected.")


# ============================================================
# TAKE OFF
# ============================================================

print()
print("Taking off...")

drone.take_off()

time.sleep(2)

print("Drone is flying.")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open camera.")

    try:
        drone.send_rc_control(0, 0, 0, 0)
        drone.land()
    except Exception:
        pass

    raise SystemExit


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# START MESSAGE
# ============================================================

print()
print("==============================================")
print(" Mission 3 - Continuous Hand Drone Control")
print("==============================================")
print()
print("HAND LEFT")
print("    -> DRONE LEFT")
print()
print("HAND CENTER / DANGER ZONE")
print("    -> DRONE STOP")
print()
print("HAND RIGHT")
print("    -> DRONE RIGHT")
print()
print("Farther from center = faster movement")
print()
print("Press Q to land and quit.")
print()


# ============================================================
# CONTROL VARIABLES
# ============================================================

smooth_x = 0.5

last_direction = "CENTER"

last_hand_seen = time.monotonic()


# ============================================================
# MEDIAPIPE
# ============================================================

try:

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        # Use actual elapsed milliseconds.
        start_time = time.monotonic()


        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            # =================================================
            # READ CAMERA FRAME
            # =================================================

            success, frame = cap.read()


            if not success:

                print("ERROR: Could not read camera.")

                break


            # =================================================
            # MIRROR CAMERA
            # =================================================

            frame = cv2.flip(frame, 1)


            # =================================================
            # FRAME SIZE
            # =================================================

            height, width, _ = frame.shape


            # =================================================
            # CENTER
            # =================================================

            center_x = width // 2


            # =================================================
            # DEADZONE
            # =================================================

            deadzone_pixels = int(
                width * CENTER_DEADZONE
            )


            deadzone_left = (
                center_x - deadzone_pixels
            )


            deadzone_right = (
                center_x + deadzone_pixels
            )


            # =================================================
            # DRAW CENTER LINE
            # =================================================

            cv2.line(
                frame,
                (center_x, 0),
                (center_x, height),
                (255, 255, 0),
                2
            )


            # =================================================
            # DRAW DANGER ZONE
            # =================================================

            cv2.line(
                frame,
                (deadzone_left, 0),
                (deadzone_left, height),
                (0, 255, 255),
                3
            )


            cv2.line(
                frame,
                (deadzone_right, 0),
                (deadzone_right, height),
                (0, 255, 255),
                3
            )


            # =================================================
            # LABELS
            # =================================================

            cv2.putText(
                frame,
                "LEFT",
                (40, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )


            cv2.putText(
                frame,
                "DANGER / STOP",
                (
                    center_x - 100,
                    45
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                "RIGHT",
                (
                    width - 130,
                    45
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


            # =================================================
            # BGR -> RGB
            # =================================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # MEDIAPIPE IMAGE
            # =================================================

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # =================================================
            # TIMESTAMP
            # =================================================

            timestamp_ms = int(
                (
                    time.monotonic()
                    -
                    start_time
                ) * 1000
            )


            # =================================================
            # SEND FRAME TO MEDIAPIPE
            # =================================================

            landmarker.detect_async(
                mp_image,
                timestamp_ms
            )


            # =================================================
            # DEFAULT CONTROL
            # =================================================

            direction = "NO HAND"

            left_right = 0

            hand_x = None


            # =================================================
            # CHECK RESULT AGE
            # =================================================

            result_age = (
                time.monotonic()
                -
                latest_result_time
            )


            result_is_valid = (
                latest_result is not None
                and
                result_age <= RESULT_TIMEOUT
            )


            # =================================================
            # PROCESS HAND
            # =================================================

            if result_is_valid:

                if len(
                    latest_result.hand_landmarks
                ) > 0:

                    hand = (
                        latest_result.hand_landmarks[0]
                    )


                    # =================================================
                    # PALM CENTER X
                    # =================================================
                    #
                    # Use palm landmarks instead of only wrist.
                    #
                    # 0  = wrist
                    # 5  = index MCP
                    # 9  = middle MCP
                    # 13 = ring MCP
                    # 17 = pinky MCP

                    raw_x = (
                        hand[0].x
                        +
                        hand[5].x
                        +
                        hand[9].x
                        +
                        hand[13].x
                        +
                        hand[17].x
                    ) / 5.0


                    raw_y = (
                        hand[0].y
                        +
                        hand[5].y
                        +
                        hand[9].y
                        +
                        hand[13].y
                        +
                        hand[17].y
                    ) / 5.0


                    # =================================================
                    # SMOOTH POSITION
                    # =================================================
                    #
                    # This is intentionally responsive.
                    #
                    # Previous:
                    # 0.15
                    #
                    # New:
                    # 0.35
                    #
                    # If still too slow, try 0.45.

                    smooth_x = (
                        (
                            1.0
                            -
                            SMOOTHING
                        )
                        *
                        smooth_x
                        +
                        SMOOTHING
                        *
                        raw_x
                    )


                    hand_x = smooth_x

                    last_hand_seen = time.monotonic()


                    # =================================================
                    # CONVERT TO PIXELS
                    # =================================================

                    hand_pixel_x = int(
                        smooth_x * width
                    )

                    hand_pixel_y = int(
                        raw_y * height
                    )


                    # =================================================
                    # DRAW HAND POSITION
                    # =================================================

                    cv2.circle(
                        frame,
                        (
                            hand_pixel_x,
                            hand_pixel_y
                        ),
                        15,
                        (0, 255, 0),
                        -1
                    )


                    # =================================================
                    # DISPLAY HAND X
                    # =================================================

                    cv2.putText(
                        frame,
                        f"Hand X: {smooth_x:.2f}",
                        (
                            20,
                            height - 115
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )


                    # =================================================
                    # CALCULATE OFFSET
                    # =================================================

                    offset = (
                        smooth_x - 0.5
                    )


                    # =================================================
                    # LEFT SIDE
                    # =================================================

                    if smooth_x < (
                        0.5
                        -
                        CENTER_DEADZONE
                    ):

                        direction = "LEFT"


                        # Distance from center.

                        distance = abs(
                            offset
                        )


                        # Convert distance to speed ratio.

                        speed_ratio = (
                            distance
                            /
                            MAX_HAND_DISTANCE
                        )


                        # Keep 0 -> 1.

                        speed_ratio = max(
                            0.0,
                            min(
                                1.0,
                                speed_ratio
                            )
                        )


                        # =================================================
                        # SPEED
                        # =================================================
                        #
                        # Just outside deadzone:
                        #     MIN_SIDE_SPEED
                        #
                        # Far left:
                        #     MAX_SIDE_SPEED

                        side_speed = int(
                            MIN_SIDE_SPEED
                            +
                            (
                                MAX_SIDE_SPEED
                                -
                                MIN_SIDE_SPEED
                            )
                            *
                            speed_ratio
                        )


                        # =================================================
                        # DRONE LEFT
                        # =================================================

                        left_right = (
                            LEFT_DIRECTION
                            *
                            side_speed
                        )


                    # =================================================
                    # RIGHT SIDE
                    # =================================================

                    elif smooth_x > (
                        0.5
                        +
                        CENTER_DEADZONE
                    ):

                        direction = "RIGHT"


                        distance = abs(
                            offset
                        )


                        speed_ratio = (
                            distance
                            /
                            MAX_HAND_DISTANCE
                        )


                        speed_ratio = max(
                            0.0,
                            min(
                                1.0,
                                speed_ratio
                            )
                        )


                        # =================================================
                        # SPEED
                        # =================================================

                        side_speed = int(
                            MIN_SIDE_SPEED
                            +
                            (
                                MAX_SIDE_SPEED
                                -
                                MIN_SIDE_SPEED
                            )
                            *
                            speed_ratio
                        )


                        # =================================================
                        # DRONE RIGHT
                        # =================================================

                        left_right = (
                            RIGHT_DIRECTION
                            *
                            side_speed
                        )


                    # =================================================
                    # CENTER DANGER ZONE
                    # =================================================

                    else:

                        direction = "CENTER / STOP"

                        left_right = 0


                else:

                    # No hand detected.

                    direction = "NO HAND"

                    left_right = 0


            else:

                # =================================================
                # RESULT TOO OLD
                # =================================================
                #
                # Stop instead of using an old position.

                direction = "NO HAND"

                left_right = 0


            # =================================================
            # SEND RC COMMAND
            # =================================================
            #
            # IMPORTANT:
            #
            # Do NOT use a large RC interval here.
            #
            # The command is sent every loop so the drone
            # continuously receives the latest hand position.

            drone.send_rc_control(
                left_right,
                FORWARD_BACKWARD,
                UP_DOWN,
                YAW
            )


            # =================================================
            # DISPLAY COLOR
            # =================================================

            if direction == "LEFT":

                color = (255, 0, 0)

            elif direction == "RIGHT":

                color = (0, 0, 255)

            elif direction == "CENTER / STOP":

                color = (0, 255, 255)

            else:

                color = (255, 255, 255)


            # =================================================
            # DISPLAY DIRECTION
            # =================================================

            cv2.putText(
                frame,
                direction,
                (
                    20,
                    height - 75
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.15,
                color,
                3
            )


            # =================================================
            # DISPLAY RC SPEED
            # =================================================

            cv2.putText(
                frame,
                f"LR SPEED: {left_right}",
                (
                    20,
                    height - 40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )


            # =================================================
            # DISPLAY SETTINGS
            # =================================================

            cv2.putText(
                frame,
                f"MAX: {MAX_SIDE_SPEED}",
                (
                    width - 150,
                    height - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # =================================================
            # SHOW CAMERA
            # =================================================

            cv2.imshow(
                "Mission 3 - Hand Gesture Drone",
                frame
            )


            # =================================================
            # KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF


            # =================================================
            # QUIT
            # =================================================

            if key == ord("q"):

                break


finally:

    # =========================================================
    # STOP DRONE
    # =========================================================

    print()
    print("Stopping drone movement...")


    try:

        drone.send_rc_control(
            0,
            0,
            0,
            0
        )

        time.sleep(0.5)

    except Exception as e:

        print(
            "Could not stop drone:",
            e
        )


    # =========================================================
    # LAND
    # =========================================================

    print("Landing drone...")


    try:

        drone.land()

        time.sleep(1)

    except Exception as e:

        print(
            "Could not land drone:",
            e
        )


    # =========================================================
    # CAMERA CLEANUP
    # =========================================================

    cap.release()

    cv2.destroyAllWindows()


    print()
    print("Mission 3 stopped.")
