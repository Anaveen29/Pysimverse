from pysimverse import Drone
import time
import keyboard

drone = Drone()
drone.connect()
drone.take_off()

# Speed settings
SPEED = 500
ROTATION_SPEED = 5

print("Drone keyboard control started!")
print("W = Forward")
print("S = Backward")
print("A = Left")
print("D = Right")
print("Up Arrow = Up")
print("Down Arrow = Down")
print("Left Arrow = Rotate Left")
print("Right Arrow = Rotate Right")
print("Q = Land and Quit")

try:
    while True:
        # Default values
        left_right = 0
        forward_backward = 0
        up_down = 0
        yaw = 0

        # Movement
        if keyboard.is_pressed("a"):
            left_right = -SPEED

        if keyboard.is_pressed("d"):
            left_right = SPEED

        if keyboard.is_pressed("w"):
            forward_backward = SPEED

        if keyboard.is_pressed("s"):
            forward_backward = -SPEED

        # Up / Down
        if keyboard.is_pressed("up"):
            up_down = SPEED

        if keyboard.is_pressed("down"):
            up_down = -SPEED

        # Rotation
        if keyboard.is_pressed("left"):
            yaw = -ROTATION_SPEED

        if keyboard.is_pressed("right"):
            yaw = ROTATION_SPEED

        # Send control commands
        drone.send_rc_control(
            left_right,
            forward_backward,
            up_down,
            yaw
        )

        # Land and quit
        if keyboard.is_pressed("q"):
            break

        time.sleep(0.05)

finally:
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)

    drone.land()
    time.sleep(1)

    print("Drone landed. Program ended.")
