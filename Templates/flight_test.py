from pysimverse import Drone
import time
drone = Drone()
drone.connect()
drone.take_off()

drone.move_forward(250)
drone.move_right(240)

# drone.rotate(70)

# drone.move_forward(20)
# time.sleep(5)
# drone.move_backward(30)
# time.sleep(5)
# drone.land()

# drone.move_forward(150)
# drone.move_left(100)
# drone.return_home()