from pysimverse import Drone
import time
import keyboard
import cv2
import os


# -----------------------------
# Drone setup
# -----------------------------

drone = Drone()
drone.connect()
time.sleep(1)

drone.streamon()
drone.take_off()


# -----------------------------
# Speed settings
# -----------------------------

SPEED = 500
ROTATION_SPEED = 5


# -----------------------------
# Create image directory
# -----------------------------

IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "captured_images"
)

os.makedirs(IMAGE_DIR, exist_ok=True)

print("Images will be saved to:")
print(IMAGE_DIR)


image_count = 0
z_was_pressed = False


# -----------------------------
# Instructions
# -----------------------------

print()
print("Drone keyboard control started!")
print()
print("W = Forward")
print("S = Backward")
print("A = Left")
print("D = Right")
print("Up Arrow = Up")
print("Down Arrow = Down")
print("Left Arrow = Rotate Left")
print("Right Arrow = Rotate Right")
print("Z = Capture Image")
print("Q = Land and Quit")
print()


# -----------------------------
# Main control loop
# -----------------------------

try:

    while True:

        # -----------------------------
        # Default movement values
        # -----------------------------

        left_right = 0
        forward_backward = 0
        up_down = 0
        yaw = 0


        # -----------------------------
        # Movement controls
        # -----------------------------

        if keyboard.is_pressed("a"):
            left_right = -SPEED

        if keyboard.is_pressed("d"):
            left_right = SPEED

        if keyboard.is_pressed("w"):
            forward_backward = SPEED

        if keyboard.is_pressed("s"):
            forward_backward = -SPEED


        # -----------------------------
        # Up / Down controls
        # -----------------------------

        if keyboard.is_pressed("up"):
            up_down = SPEED

        if keyboard.is_pressed("down"):
            up_down = -SPEED


        # -----------------------------
        # Rotation controls
        # -----------------------------

        if keyboard.is_pressed("left"):
            yaw = -ROTATION_SPEED

        if keyboard.is_pressed("right"):
            yaw = ROTATION_SPEED


        # -----------------------------
        # Send movement commands
        # -----------------------------

        drone.send_rc_control(
            left_right,
            forward_backward,
            up_down,
            yaw
        )


        # -----------------------------
        # Get camera frame
        # -----------------------------

        frame, is_success = drone.get_frame()


        if is_success:

            # Show live drone footage
            cv2.imshow("Drone Camera", frame)


            # -----------------------------
            # Z = Capture image
            # -----------------------------

            z_pressed = keyboard.is_pressed("z")

            # Capture only when Z changes
            # from NOT pressed -> pressed
            if z_pressed and not z_was_pressed:

                image_count += 1

                filename = os.path.join(
                    IMAGE_DIR,
                    f"image_{image_count}.jpg"
                )

                success = cv2.imwrite(filename, frame)

                if success:
                    print()
                    print("Image captured successfully!")
                    print(f"Saved to: {filename}")
                else:
                    print()
                    print("ERROR: Image could not be saved.")
                    print(f"Path: {filename}")

            # Remember current Z state
            z_was_pressed = z_pressed


        else:
            print("Camera frame not received.")


        # -----------------------------
        # Q = Land and quit
        # -----------------------------

        if keyboard.is_pressed("q"):
            break


        # OpenCV event handling
        cv2.waitKey(1)

        time.sleep(0.05)


# -----------------------------
# Safety / Cleanup
# -----------------------------

finally:

    drone.send_rc_control(0, 0, 0, 0)

    time.sleep(0.5)

    drone.land()

    time.sleep(1)

    cv2.destroyAllWindows()

    print()
    print("Drone landed.")
    print(f"Images saved in: {IMAGE_DIR}")
    print("Program ended.")
