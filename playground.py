import pygame
import rospy

from std_msgs.msg import Int32

class ROSPackage_PS4Joystick:
    def __init__(self):
        self.pygame.init()
        self.j = pygame.joystick.Joystick(0)
        self.j.init()
        
    def start(self):
        rosppy.init_node('ps4_joystick')
        
        pub_steering = rospy.Publisher('steering', Int32, queue_size = 10)
        pub_throttle = rospy.Publisher('throttle', Int32, queue_size = 10)

        rate = rospy.Rate(60)

        while not rospy.is_shutdown():
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.JOYAXISMOTION:
                    if event.axis == 5:
                        pub_steering(axis.value)
                    else if event.axis == 1:
                        pub_throttle(event)
                    # maybe we need to put is somewhere else
                    rate.sleep()

    def stop(self)
        self.j.quit()


if __name__ == '__main__':
    package = ROSPackage_PS4Joystick()

    try:
        package.start()
    except rospy.ROSPackage_PS4Joystick:
        package.stop()
