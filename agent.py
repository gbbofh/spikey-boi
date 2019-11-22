import network
import numpy as np
import scipy.stats as stat

import math

from turtle import Screen
from turtle import Turtle

class Agent():

    def __init__(self, ne=0, ni=0):

        # Initialize neural network, target position, and distance
        self.net = network.Network(ne, ni, 3, 2)
        self.left_wheel = 0.0
        self.right_wheel = 0.0
        self.target_x = np.random.randint(-390, 390)
        self.target_y = np.random.randint(-240, 240)
        self.target_dist = math.sqrt(self.target_x ** 2 + self.target_y ** 2)

        self.ne = ne
        self.ni = ni

        # Create a graphic for the agent
        self.gfx = Turtle()
        self.gfx.speed(0)
        self.gfx.shape('square')
        self.gfx.color('red')
        self.gfx.goto(0, 0)

        # Create a graphic for the target
        # All this code should get ripped out into its own class
        # eventually.
        self.target_gfx = Turtle()
        self.target_gfx.speed(0)
        self.target_gfx.shape('circle')
        self.target_gfx.color('green')
        self.target_gfx.penup()
        self.target_gfx.goto(self.target_x, self.target_y)

        # Used for testing -- now defunct
        self.motor_frequency = np.zeros(2)
        self.motor_accum_window = 1


    def update(self):
        # Generate stochastic inputs for all neurons in the network
        self.net.input[0 : self.net.numEx] = 5.0 * stat.norm.rvs(size=self.net.numEx)
        self.net.input[self.net.numEx : ] = 2.0 * stat.norm.rvs(size=self.net.numIn)

        # Find the distance to the target
        dX = self.target_x - self.gfx.xcor()
        dY = self.target_y - self.gfx.ycor()
        self.target_dist = math.sqrt(dX ** 2 + dY ** 2)

        # We want the sensory input to the agent to increase as it draws nearer
        # to the target -- 450 was determined such that the smallest input is
        # approximately zero, and the largest input is approximately 10 when
        # the agent is right next to the target
        self.net.sensoryInput[0] = min(450.0 / self.target_dist, 10.0)

        # Determine the angle between the direction the agent is facing
        # and the target
        angle = math.atan2(dY, dX)
        heading = self.gfx.heading() * math.pi / 180
        heading = heading if heading <= math.pi else heading - math.pi
        if abs(angle) > abs(heading):
            angle = angle - heading
        else:
            angle = heading - angle

        self.net.sensoryInput[1] = 2 * angle
        self.net.sensoryInput[2] = -2 * angle

        # Update the network with the newly generated inputs
        self.net.update()

        # TESTING SPIKE TIMING -> SPIKE FREQ
        # DID NOT WORK VERY WELL

        # mo = np.where(self.net.voltage[self.net.motorSyn] >= 30.0)
        # self.motor_frequency[mo] += 1

        # self.motor_accum_window += 1

        # if self.motor_accum_window % 100:
        #     self.left_wheel = self.motor_frequency[0] / self.motor_accum_window
        #     self.right_wheel = self.motor_frequency[1] / self.motor_accum_window
        #     self.motor_frequency *= 0

        # END TESTING

        # Feed the output of the motor neurons into the "motors" for our agent
        self.left_wheel = self.net.motorOutput[0] * 5
        self.right_wheel = self.net.motorOutput[1] * 5

        # L > R -> turn right
        # L < R -> turn left
        # L == R -> go forward
        self.gfx.right((self.left_wheel - self.right_wheel) * 2.0)
        self.gfx.forward((self.left_wheel + self.right_wheel))


def main():
    win = Screen()
    win.bgcolor('black')
    win.setup(width=800,height=600)
    win.tracer(0, 0)

    # Debug drawing for neuron spikes
    sD = Turtle()
    sD.speed(0)
    sD.shape('square')
    sD.color((1.0, 1.0, 1.0))
    sD.penup()
    sD.goto(-390, 290)
    sD.hideturtle()

    # Debug drawing for synaptic matrix
    synD = Turtle()
    synD.speed(0)
    synD.shape('square')
    synD.penup()
    synD.goto(-390, 290)
    synD.hideturtle()

    a = Agent(15, 8)

    # I hate this, but it works well and keeps the sim from lagging too much
    def draw_synapses():
        synD.clearstamps()
        synD.goto(-390, 290)
        for s in a.net.synapses:
            for e in s:
                color = (float(e), float(e), float(e)) if e > 0 else (float(-e), 0, 0)
                synD.color(color)
                synD.stamp()
                synD.goto(synD.xcor() + 20, synD.ycor())
            synD.goto(-390, synD.ycor() - 20)

        if draw_synapses:
            win.ontimer(draw_synapses, 100)

    def set_target(x, y):
        a.target_x = x
        a.target_y = y
        a.target_gfx.goto(a.target_x, a.target_y)

    # UNCOMMENT THIS LINE TO ENABLE DRAWING
    # FOR THE SYNAPTIC WEIGHTS
    # win.ontimer(draw_synapses, 0)
    win.onclick(set_target)

    while True:
        a.update()
        xpos = a.gfx.xcor()
        ypos = a.gfx.ycor()
        if xpos > 390 or xpos < -390:
            a.gfx.goto(-(xpos + 20 * np.sign(-xpos)), ypos)
            a.gfx.clear()
        if ypos > 290 or ypos < -290:
            a.gfx.goto(xpos, -(ypos + 20 * np.sign(-ypos)))
            a.gfx.clear()

        if a.target_dist < 20:
            print('Spikey bob found the target.')
            a.target_x = np.random.randint(-390, 390)
            a.target_y = np.random.randint(-240, 240)
            a.target_dist = math.sqrt(a.target_x ** 2 + a.target_y ** 2)
            a.target_gfx.goto(a.target_x, a.target_y)

            # Attempt at reinforcing on consumption of food
            a.net.synapses = np.clip(a.net.synapses * 1.1, -1.0, 1.0)

        sD.clearstamps()
        sD.goto(-390, -285)
        for i,v in enumerate(a.net.voltage):
            nv = (v + 65) / 95
            nv = min(max(0.0, nv), 1.0)
            color = (float(nv), float(nv), float(nv))
            if i in a.net.sensorySyn:
                color = (0.0, float(nv), 0.0)
            elif i in a.net.motorSyn:
                color = (float(nv), 0.0, float(nv / 2.0))
            sD.color(color)
            sD.stamp()
            sD.goto(sD.xcor() + 20, -285)


        win.update()


if __name__ == '__main__':
    main()
