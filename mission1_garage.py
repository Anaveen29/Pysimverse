from pysimverse import Drone
import time
drone = Drone()
drone.connect()
drone.take_off()
drone.speed = 900

# Garage 1 Mission

drone.move_forward(250)
drone.move_right(240)

# Garage 2 Mission

# drone.move_left(200)
# drone.move_forward(190)
# drone.move_right(430)
# drone.move_forward(120)

# Garage 3 Mission

# drone.move_left(230)
# drone.move_forward(340)
# drone.move_right(500)
# drone.move_down(100)
# drone.move_backward(180)



drone.land()
time.sleep(1)