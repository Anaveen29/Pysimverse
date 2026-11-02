import cv2
import mediapipe as mp
import os
import time

from pysimverse import Drone


# ============================================================
# MISSION 4
# JUMP DETECTION -> DRONE MOVE UP
# ============================================================


# ============================================================
# MODEL PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "pose_landmarker.task"
)


# ============================================================
# MODEL CHECK
# ============================================================

print()
print("==============================================")
print(" Checking Pose Landmarker model")
print("==============================================")
print()

print("Project directory:")
print(PROJECT_DIR)

print("Model path:")
print(MODEL_PATH)
print()


if not os.path.isfile(MODEL_PATH):

    print("ERROR: Pose Landmarker model not found!")
    print()
    print("Expected:")
    print(MODEL_PATH)
    print()

    raise SystemExit


model_size = os.path.getsize(MODEL_PATH)

if model_size <= 0:

    print("ERROR: pose_landmarker.task is empty!")

    raise SystemExit


print("Pose Landmarker model found!")
print(f"Model size: {model_size:,} bytes")
print()


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

PoseLandmarkerResult = (
    mp.tasks.vision.PoseLandmarkerResult
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# LATEST POSE RESULT
# ============================================================

latest_result = None
latest_result_time = 0.0


# ============================================================
# MEDIAPIPE CALLBACK
# ============================================================

def pose_result_callback(
    result: PoseLandmarkerResult,
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

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.LIVE_STREAM,

    num_poses=1,

    min_pose_detection_confidence=0.60,

    min_pose_presence_confidence=0.60,

    min_tracking_confidence=0.60,

    result_callback=pose_result_callback
)


# ============================================================
# JUMP DETECTION SETTINGS
# ============================================================

UPWARD_VELOCITY_THRESHOLD = 0.14

DOWNWARD_VELOCITY_THRESHOLD = -0.05

MIN_UPWARD_DISPLACEMENT = 0.07

MIN_AIR_TIME = 0.20

MAX_JUMP_TIME = 2.0

LANDING_TOLERANCE = 0.10

JUMP_COOLDOWN = 0.80

MAX_RESULT_AGE = 0.30

VELOCITY_SMOOTHING = 0.30


# ============================================================
# DRONE SETTINGS
# ============================================================

# send_rc_control:
#
#     left_right
#     forward_backward
#     up_down
#     yaw
#
# Positive up_down is assumed to mean UP,
# based on the PySimverse interface you are using.
#

DRONE_UP_VALUE = 50


# How long the drone should receive the UP command.

DRONE_UP_DURATION = 1.00


# RC commands should be refreshed continuously.

DRONE_COMMAND_INTERVAL = 0.05


# ============================================================
# HIP LANDMARKS
# ============================================================

LEFT_HIP = 23
RIGHT_HIP = 24


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open camera.")

    raise SystemExit


cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# DRONE VARIABLES
# ============================================================

drone = None

drone_up_until = 0.0

last_drone_command_time = 0.0

drone_is_moving_up = False


# ============================================================
# JUMP STATES
# ============================================================

GROUND = "GROUND"

TAKEOFF = "TAKEOFF"

AIRBORNE = "AIRBORNE"

LANDING = "LANDING"

jump_state = GROUND


# ============================================================
# JUMP DATA
# ============================================================

takeoff_hip_y = None

highest_hip_y = None

takeoff_time = None

last_jump_time = -999.0


# ============================================================
# JUMP COUNTER
# ============================================================

jump_count = 0


# ============================================================
# HIP HISTORY
# ============================================================

previous_hip_y = None

previous_time = None


# ============================================================
# VELOCITY
# ============================================================

raw_velocity = 0.0

smooth_velocity = 0.0


# ============================================================
# SEND DRONE UP
# ============================================================

def send_drone_up(current_time):

    global last_drone_command_time
    global drone_is_moving_up

    if drone is None:

        return False


    # --------------------------------------------------------
    # Refresh command only every 50 ms.
    # --------------------------------------------------------

    if (
        current_time
        -
        last_drone_command_time
        <
        DRONE_COMMAND_INTERVAL
    ):

        return True


    try:

        drone.send_rc_control(
            0,
            0,
            DRONE_UP_VALUE,
            0
        )

        last_drone_command_time = current_time

        if not drone_is_moving_up:

            print(
                f"[DRONE] UP command started: "
                f"up_down={DRONE_UP_VALUE}"
            )

            drone_is_moving_up = True

        return True


    except Exception as e:

        print()
        print(
            "[DRONE ERROR] send_rc_control() failed:"
        )
        print(e)
        print()

        return False


# ============================================================
# STOP DRONE MOVEMENT
# ============================================================

def stop_drone(current_time):

    global last_drone_command_time
    global drone_is_moving_up

    if drone is None:

        return False


    # --------------------------------------------------------
    # Do not spam stop command.
    # --------------------------------------------------------

    if (
        current_time
        -
        last_drone_command_time
        <
        DRONE_COMMAND_INTERVAL
    ):

        return True


    try:

        drone.send_rc_control(
            0,
            0,
            0,
            0
        )

        last_drone_command_time = current_time

        if drone_is_moving_up:

            print(
                "[DRONE] UP command stopped."
            )

            drone_is_moving_up = False

        return True


    except Exception as e:

        print()
        print(
            "[DRONE ERROR] Stop command failed:"
        )
        print(e)
        print()

        return False


# ============================================================
# START DRONE UP MOVEMENT
# ============================================================

def start_drone_up(current_time):

    global drone_up_until

    if drone is None:

        print(
            "[DRONE ERROR] Drone object is None."
        )

        return False


    drone_up_until = (
        current_time
        +
        DRONE_UP_DURATION
    )


    print()
    print("==============================================")
    print("          DRONE UP TRIGGERED")
    print("==============================================")
    print(
        f"UP VALUE      : {DRONE_UP_VALUE}"
    )
    print(
        f"UP DURATION   : {DRONE_UP_DURATION:.2f} sec"
    )
    print(
        f"UP UNTIL      : {drone_up_until:.3f}"
    )
    print("==============================================")
    print()


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Send UP immediately.
    # --------------------------------------------------------

    try:

        drone.send_rc_control(
            0,
            0,
            DRONE_UP_VALUE,
            0
        )

        print(
            "[DRONE] Immediate UP command sent."
        )

        return True


    except Exception as e:

        print()
        print(
            "[DRONE ERROR] Immediate UP failed:"
        )
        print(e)
        print()

        return False


# ============================================================
# DRONE INITIALIZATION
# ============================================================

try:

    print()
    print("==============================================")
    print(" Connecting to PySimverse drone")
    print("==============================================")
    print()


    drone = Drone()


    print(
        "Calling drone.connect()..."
    )

    drone.connect()


    print(
        "Drone connected."
    )

    print()


    print(
        "Calling drone.take_off()..."
    )

    drone.take_off()


    print(
        "Drone takeoff complete."
    )

    print()


    # Give simulator time to settle.

    time.sleep(1.0)


    # --------------------------------------------------------
    # IMPORTANT TEST
    # --------------------------------------------------------
    #
    # Do NOT send UP here.
    #
    # UP should only happen after a confirmed jump.
    #

    print(
        "Drone is ready."
    )

    print()


except Exception as e:

    print()
    print(
        "ERROR: Could not initialize drone."
    )

    print(e)
    print()

    cap.release()

    cv2.destroyAllWindows()

    raise SystemExit


# ============================================================
# MAIN PROGRAM
# ============================================================

try:

    with PoseLandmarker.create_from_options(
        options
    ) as landmarker:


        start_time = time.monotonic()


        print("==============================================")
        print(" Mission 4 Started")
        print(" Jump Detection -> Drone UP")
        print("==============================================")
        print()

        print(
            "DRONE CONTROL:"
        )

        print(
            "  send_rc_control(0, 0, 50, 0)"
        )

        print()

        print(
            "Press Q to quit."
        )

        print()


        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:


            # =================================================
            # CURRENT TIME
            # =================================================

            current_time = time.monotonic()


            # =================================================
            # DRONE CONTROL
            # =================================================
            #
            # THIS IS THE IMPORTANT PART.
            #
            # While the UP timer is active:
            #
            #     send_rc_control(0, 0, 50, 0)
            #
            # repeatedly.
            #
            # When timer expires:
            #
            #     send_rc_control(0, 0, 0, 0)
            #

            if (
                current_time
                <
                drone_up_until
            ):

                send_drone_up(
                    current_time
                )

            else:

                stop_drone(
                    current_time
                )


            # =================================================
            # CAMERA
            # =================================================

            success, frame = cap.read()


            if not success:

                print(
                    "ERROR: Could not read camera frame."
                )

                break


            # =================================================
            # MIRROR
            # =================================================

            frame = cv2.flip(
                frame,
                1
            )


            height, width, _ = frame.shape


            # =================================================
            # MEDIAPIPE IMAGE
            # =================================================

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # =================================================
            # TIMESTAMP
            # =================================================

            timestamp_ms = int(
                (
                    current_time
                    -
                    start_time
                )
                * 1000
            )


            # =================================================
            # SEND FRAME TO MEDIAPIPE
            # =================================================

            landmarker.detect_async(
                mp_image,
                timestamp_ms
            )


            # =================================================
            # DEFAULTS
            # =================================================

            hip_x = None

            hip_y = None

            raw_velocity = 0.0

            jump_detected_now = False

            state_text = "WAITING"


            # =================================================
            # CHECK RESULT
            # =================================================

            result_is_fresh = (

                latest_result is not None

                and

                (
                    current_time
                    -
                    latest_result_time
                )
                <= MAX_RESULT_AGE
            )


            # =================================================
            # PROCESS POSE
            # =================================================

            if result_is_fresh:


                if (
                    len(
                        latest_result.pose_landmarks
                    )
                    > 0
                ):


                    pose = (
                        latest_result.pose_landmarks[0]
                    )


                    # =================================================
                    # HIPS
                    # =================================================

                    left_hip = pose[
                        LEFT_HIP
                    ]

                    right_hip = pose[
                        RIGHT_HIP
                    ]


                    # =================================================
                    # VISIBILITY
                    # =================================================

                    left_visibility = getattr(
                        left_hip,
                        "visibility",
                        1.0
                    )

                    right_visibility = getattr(
                        right_hip,
                        "visibility",
                        1.0
                    )


                    if (
                        left_visibility >= 0.40

                        and

                        right_visibility >= 0.40
                    ):


                        # =================================================
                        # HIP CENTER
                        # =================================================

                        hip_x = (
                            left_hip.x
                            +
                            right_hip.x
                        ) / 2.0


                        hip_y = (
                            left_hip.y
                            +
                            right_hip.y
                        ) / 2.0


                        # =================================================
                        # DRAW HIPS
                        # =================================================

                        left_pixel = (
                            int(
                                left_hip.x
                                *
                                width
                            ),
                            int(
                                left_hip.y
                                *
                                height
                            )
                        )


                        right_pixel = (
                            int(
                                right_hip.x
                                *
                                width
                            ),
                            int(
                                right_hip.y
                                *
                                height
                            )
                        )


                        center_pixel = (
                            int(
                                hip_x
                                *
                                width
                            ),
                            int(
                                hip_y
                                *
                                height
                            )
                        )


                        cv2.circle(
                            frame,
                            left_pixel,
                            7,
                            (255, 0, 0),
                            -1
                        )


                        cv2.circle(
                            frame,
                            right_pixel,
                            7,
                            (0, 0, 255),
                            -1
                        )


                        cv2.circle(
                            frame,
                            center_pixel,
                            11,
                            (0, 255, 0),
                            -1
                        )


                        # =================================================
                        # VELOCITY
                        # =================================================

                        if (
                            previous_hip_y
                            is not None

                            and

                            previous_time
                            is not None
                        ):


                            dt = (
                                current_time
                                -
                                previous_time
                            )


                            if (
                                dt > 0.005

                                and

                                dt < 0.5
                            ):


                                raw_velocity = (
                                    previous_hip_y
                                    -
                                    hip_y
                                ) / dt


                                raw_velocity = max(
                                    -3.0,
                                    min(
                                        3.0,
                                        raw_velocity
                                    )
                                )


                                smooth_velocity = (
                                    (
                                        1.0
                                        -
                                        VELOCITY_SMOOTHING
                                    )
                                    *
                                    smooth_velocity
                                    +
                                    VELOCITY_SMOOTHING
                                    *
                                    raw_velocity
                                )


                        # =================================================
                        # SAVE HISTORY
                        # =================================================

                        previous_hip_y = hip_y

                        previous_time = current_time


                        # =================================================
                        # GROUND
                        # =================================================

                        if jump_state == GROUND:


                            state_text = "GROUND"


                            if (
                                smooth_velocity
                                >=
                                UPWARD_VELOCITY_THRESHOLD

                                and

                                (
                                    current_time
                                    -
                                    last_jump_time
                                )
                                >=
                                JUMP_COOLDOWN
                            ):


                                takeoff_hip_y = hip_y

                                highest_hip_y = hip_y

                                takeoff_time = current_time

                                jump_state = TAKEOFF


                                print(
                                    "Takeoff detected..."
                                )


                        # =================================================
                        # TAKEOFF
                        # =================================================

                        elif jump_state == TAKEOFF:


                            state_text = "TAKEOFF"


                            if (
                                highest_hip_y
                                is None

                                or

                                hip_y
                                <
                                highest_hip_y
                            ):

                                highest_hip_y = hip_y


                            upward_displacement = (
                                takeoff_hip_y
                                -
                                hip_y
                            )


                            if (
                                upward_displacement
                                >=
                                MIN_UPWARD_DISPLACEMENT
                            ):


                                jump_state = AIRBORNE


                                print(
                                    "Airborne detected..."
                                )


                            elif (
                                (
                                    current_time
                                    -
                                    takeoff_time
                                )
                                >
                                0.60

                                and

                                smooth_velocity
                                <
                                0.02
                            ):


                                jump_state = GROUND

                                takeoff_hip_y = None

                                highest_hip_y = None

                                takeoff_time = None


                        # =================================================
                        # AIRBORNE
                        # =================================================

                        elif jump_state == AIRBORNE:


                            state_text = "AIRBORNE"


                            if (
                                highest_hip_y
                                is None

                                or

                                hip_y
                                <
                                highest_hip_y
                            ):

                                highest_hip_y = hip_y


                            air_time = (
                                current_time
                                -
                                takeoff_time
                            )


                            if (
                                air_time
                                >=
                                MIN_AIR_TIME

                                and

                                smooth_velocity
                                <=
                                DOWNWARD_VELOCITY_THRESHOLD
                            ):


                                jump_state = LANDING


                                print(
                                    "Downward movement detected..."
                                )


                            elif (
                                air_time
                                >
                                MAX_JUMP_TIME
                            ):


                                jump_state = GROUND

                                takeoff_hip_y = None

                                highest_hip_y = None

                                takeoff_time = None


                        # =================================================
                        # LANDING
                        # =================================================

                        elif jump_state == LANDING:


                            state_text = "LANDING"


                            landing_difference = abs(
                                hip_y
                                -
                                takeoff_hip_y
                            )


                            if (
                                landing_difference
                                <=
                                LANDING_TOLERANCE
                            ):


                                if (
                                    (
                                        current_time
                                        -
                                        last_jump_time
                                    )
                                    >=
                                    JUMP_COOLDOWN
                                ):


                                    jump_count += 1

                                    jump_detected_now = True

                                    last_jump_time = current_time


                                # -----------------------------------------
                                # RESET
                                # -----------------------------------------

                                jump_state = GROUND

                                takeoff_hip_y = None

                                highest_hip_y = None

                                takeoff_time = None


                        # =================================================
                        # JUMP CONFIRMED
                        # =================================================

                        if jump_detected_now:


                            print()
                            print("==============================================")
                            print("             JUMP DETECTED!")
                            print(
                                f"             JUMP COUNT = {jump_count}"
                            )
                            print("             DRONE -> UP")
                            print(
                                f"             UP VALUE = {DRONE_UP_VALUE}"
                            )
                            print(
                                f"             UP TIME = "
                                f"{DRONE_UP_DURATION:.2f}s"
                            )
                            print("==============================================")
                            print()


                            # =================================================
                            # START UP MOVEMENT
                            # =================================================

                            move_success = (
                                start_drone_up(
                                    current_time
                                )
                            )


                            if move_success:

                                state_text = (
                                    "JUMP -> DRONE UP"
                                )

                            else:

                                state_text = (
                                    "JUMP -> DRONE ERROR"
                                )


                        # =================================================
                        # DISPLAY HIP
                        # =================================================

                        cv2.putText(
                            frame,
                            f"Hip Y: {hip_y:.3f}",
                            (
                                20,
                                height - 145
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.60,
                            (255, 255, 255),
                            2
                        )


                        # =================================================
                        # DISPLAY VELOCITY
                        # =================================================

                        cv2.putText(
                            frame,
                            (
                                f"Velocity: "
                                f"{smooth_velocity:.3f}"
                            ),
                            (
                                20,
                                height - 115
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.60,
                            (0, 255, 255),
                            2
                        )


                        # =================================================
                        # DISPLAY DISPLACEMENT
                        # =================================================

                        if (
                            takeoff_hip_y
                            is not None
                        ):

                            displacement = (
                                takeoff_hip_y
                                -
                                hip_y
                            )

                        else:

                            displacement = 0.0


                        cv2.putText(
                            frame,
                            (
                                f"Up displacement: "
                                f"{displacement:.3f}"
                            ),
                            (
                                20,
                                height - 85
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 200, 0),
                            2
                        )


                        # =================================================
                        # STATE
                        # =================================================

                        cv2.putText(
                            frame,
                            f"State: {state_text}",
                            (
                                20,
                                height - 55
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2
                        )


                        # =================================================
                        # HIP
                        # =================================================

                        cv2.putText(
                            frame,
                            (
                                f"Hip: "
                                f"({hip_x:.2f}, "
                                f"{hip_y:.2f})"
                            ),
                            (
                                20,
                                height - 25
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (255, 255, 255),
                            1
                        )


                    else:

                        state_text = (
                            "LOW HIP CONFIDENCE"
                        )

                        previous_hip_y = None

                        previous_time = None

                        smooth_velocity = 0.0


                else:

                    state_text = "NO POSE"

                    previous_hip_y = None

                    previous_time = None

                    smooth_velocity = 0.0

                    jump_state = GROUND

                    takeoff_hip_y = None

                    highest_hip_y = None

                    takeoff_time = None


            else:

                state_text = (
                    "WAITING FOR POSE"
                )

                previous_hip_y = None

                previous_time = None

                smooth_velocity = 0.0


            # =================================================
            # DRONE STATUS
            # =================================================

            if (
                current_time
                <
                drone_up_until
            ):

                drone_status = (
                    "DRONE: MOVING UP"
                )

                drone_color = (
                    0,
                    165,
                    255
                )

            else:

                drone_status = (
                    "DRONE: READY"
                )

                drone_color = (
                    255,
                    255,
                    255
                )


            # =================================================
            # STATUS COLOR
            # =================================================

            if state_text == "NO POSE":

                status_color = (
                    0,
                    0,
                    255
                )

            elif state_text == "WAITING FOR POSE":

                status_color = (
                    0,
                    255,
                    255
                )

            elif state_text == "AIRBORNE":

                status_color = (
                    255,
                    0,
                    255
                )

            elif (
                state_text
                ==
                "JUMP -> DRONE UP"
            ):

                status_color = (
                    0,
                    165,
                    255
                )

            else:

                status_color = (
                    0,
                    255,
                    0
                )


            # =================================================
            # TOP STATE
            # =================================================

            cv2.putText(
                frame,
                state_text,
                (
                    20,
                    40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                3
            )


            # =================================================
            # JUMP COUNTER
            # =================================================

            cv2.rectangle(
                frame,
                (
                    width - 250,
                    10
                ),
                (
                    width - 10,
                    82
                ),
                (25, 25, 25),
                -1
            )


            cv2.putText(
                frame,
                f"JUMPS: {jump_count}",
                (
                    width - 235,
                    60
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )


            # =================================================
            # DRONE STATUS
            # =================================================

            cv2.putText(
                frame,
                drone_status,
                (
                    width - 250,
                    112
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                drone_color,
                2
            )


            # =================================================
            # DEBUG VALUES
            # =================================================

            cv2.putText(
                frame,
                (
                    f"Up threshold: "
                    f"{UPWARD_VELOCITY_THRESHOLD:.2f}"
                ),
                (
                    width - 250,
                    140
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1
            )


            cv2.putText(
                frame,
                (
                    f"Min displacement: "
                    f"{MIN_UPWARD_DISPLACEMENT:.2f}"
                ),
                (
                    width - 250,
                    163
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1
            )


            cv2.putText(
                frame,
                (
                    f"UP value: "
                    f"{DRONE_UP_VALUE}"
                ),
                (
                    width - 250,
                    186
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 200, 255),
                1
            )


            cv2.putText(
                frame,
                (
                    f"UP duration: "
                    f"{DRONE_UP_DURATION:.1f}s"
                ),
                (
                    width - 250,
                    209
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 200, 255),
                1
            )


            # =================================================
            # INSTRUCTIONS
            # =================================================

            cv2.putText(
                frame,
                "Jump -> Drone UP",
                (
                    20,
                    70
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                "Q = Quit / Land",
                (
                    20,
                    98
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 200, 200),
                1
            )


            # =================================================
            # SHOW
            # =================================================

            cv2.imshow(
                "Mission 4 - Jump -> Drone UP",
                frame
            )


            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            if key == ord("q"):

                break


            # =================================================
            # SMALL DELAY
            # =================================================

            time.sleep(0.01)


finally:

    # =========================================================
    # STOP DRONE
    # =========================================================

    if drone is not None:

        try:

            print()
            print(
                "Stopping drone movement..."
            )

            drone.send_rc_control(
                0,
                0,
                0,
                0
            )

            time.sleep(0.2)

        except Exception as e:

            print(
                "Warning stopping drone:",
                e
            )


        # =====================================================
        # LAND
        # =====================================================

        try:

            print(
                "Landing drone..."
            )

            drone.land()

            print(
                "Drone landed."
            )

        except Exception as e:

            print(
                "Warning landing drone:",
                e
            )


    # =========================================================
    # CAMERA CLEANUP
    # =========================================================

    cap.release()

    cv2.destroyAllWindows()


    # =========================================================
    # FINAL RESULT
    # =========================================================

    print()
    print("==============================================")
    print(" Mission 4 stopped")
    print(
        f" Total jumps detected: {jump_count}"
    )
    print("==============================================")
