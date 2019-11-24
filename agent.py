import network
import numpy as np
import scipy.stats as stat

import matplotlib.pyplot as plot
import matplotlib.animation as animation

import math

from turtle import Screen
from turtle import Turtle

# NOTE: Upon examining behaviour on the smallest possible network I can feasibly
# see solving this problem (2 inputs -> 2 outputs), it appears that the "motor"
# neurons are being trained to drive the "sensory" neurons, instead of the other
# way around. A simple fix for this may be to separate everything into three
# layers:
#  _________________
# | i i i ... i i i | <--- unidirectional input layer
# |_________________|
# | | | | | | | | | |
#  _________________
# | h->h<->h h h h  |
# | | /|   |/ \|/   | <--- recurrent hidden layer
# | h<-h-->h<->h    |
# |_________________|
# | | | | | | | | | |
#  _________________
# | o o o ... o o o | <--- unidirectional output layer
# |_________________|

class Agent():

    def __init__(self, ne=0, ni=0):

        # Initialize neural network, target position, and distance
        # self.net = network.Network(ne, ni, 4, 2)
        self.net = network.Network(ne, ni, ne / 2,  ne / 2)
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
        # self.motor_frequency = np.zeros(2)
        # self.motor_accum_window = 1

        self.food_reward = 0.0


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
        # self.net.sensoryInput[0] = min(450.0 / self.target_dist, 10.0)

        angle = self.gfx.towards(self.target_gfx)
        heading = self.gfx.heading()
        angle = angle - heading
        angle += 180
        angle = angle if angle <= 180 else angle - 360
        angle *= math.pi / 180

        self.net.sensoryInput[0] = 2 * angle
        self.net.sensoryInput[1] = -2 * angle

        # self.net.sensoryInput[1] = 2 * angle
        # self.net.sensoryInput[2] = -2 * angle
        # self.net.sensoryInput[3] = self.food_reward

        self.food_reward -= 0.001
        self.food_reward = max(0.0, self.food_reward)

        # Update the network with the newly generated inputs
        self.net.update()

        # TESTING SPIKE TIMING -> SPIKE FREQ
        # DID NOT WORK VERY WELL

        # mo = np.where(self.net.voltage[self.net.motorSyn] >= 30.0)
        # self.motor_frequency[mo] += 1

        # self.motor_accum_window += 1

        # if self.motor_accum_window % 10:
        #     self.left_wheel = self.motor_frequency[0] / 10
        #     self.right_wheel = self.motor_frequency[1] / 10
        #     self.motor_frequency *= 0
        #     motor_accum_window = 0

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

    textD = Turtle()
    textD.speed(0)
    textD.color((1.0, 1.0, 1.0))
    textD.penup()
    textD.goto(180, -290)
    textD.hideturtle()

    #a = Agent(15, 8)
    #a = Agent(14, 6)
    a = Agent(4, 0)

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
    win.ontimer(draw_synapses, 0)
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
            a.food_reward = 10.0

            # Attempt at reinforcing on consumption of food
            a.net.synapses = np.clip(a.net.synapses * 1.1, -1.0, 1.0)

        sD.clearstamps()
        sD.goto(-390, -285)

        textD.clear()
        textD.goto(180, -290)
        angle = a.net.sensoryInput[1] * 180 / (2 * math.pi)
        textD.write('Angle: {}, {}'.format(str(angle),
                                            str(-angle)),
                                            font=('Arial', 11, 'normal'))

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
