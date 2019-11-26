import os
import time
import turtle

import tkinter
import tkinter.filedialog

from functools import partial

import numpy

from text import Text
from target import Target
from entity import Entity
from new_agent import Agent


class Application():

    rootWindow = None

    run = True
    synapseDebug = False
    controlText = '\n'.join(('Press Q to Exit',
                            'Press S to Draw Synapses',
                            'Press D to Dump Synapses to File',
                            'Press L to Load Synapses from File',
                            'Press 1-3 to Set Speed'))
    debug = None
    spikes = None
    updateDelta = 0.001
    speedChanged = False

    def registerClose():
        Application.run = False


    def enableSynapseDebug(win, net):
        Application.synapseDebug = not Application.synapseDebug
        if not Application.synapseDebug:
            Application.debug.clearstamps()


    def setSpeed(speed):
        speeds = {
                1 : 0.001,
                2 : 0.05,
                3 : 0.1
        }

        Application.speedChanged = True
        Application.updateDelta = speeds[speed]


    def writeSynapses(net):
        types = [('all files', '.*'), ('synapses', '.syn')]

        sensorySyn = net.inputSynapses
        recurrentSyn = net.synapses
        motorSyn = net.outputSynapses

        ans = tkinter.filedialog.asksaveasfilename(parent=Application.rootWindow,
                                                    initialdir=os.getcwd(),
                                                    title='Save Synapses',
                                                    filetypes=types)

        with open(ans, 'w') as fp:
            for s in sensorySyn:
                for e in s:
                    fp.write(str(e) + ' ')
                fp.write('\n')
            fp.write('\n')
            for s in recurrentSyn:
                for e in s:
                    fp.write(str(e) + ' ')
                fp.write('\n')
            fp.write('\n')
            for s in motorSyn:
                for e in s:
                    fp.write(str(e) + ' ')
                fp.write('\n')
            fp.write('\n')


    def readSynapses(net):
        types = [('all files', '.*'), ('synapses', '.syn')]

        ans = tkinter.filedialog.askopenfilename(parent=Application.rootWindow,
                                                    initialdir=os.getcwd(),
                                                    title='Load Synapses',
                                                    filetypes=types)
        with open(ans, 'r') as fp:
            senList = []
            recList = []
            motList = []

            line = None

            # I hate everything about these loops,
            # but they need to be this way
            # because assignments in loop conditions
            # are not valid in python :/
            while True:
                line = fp.readline().strip()
                if not line:
                    break
                senList.append([float(x) for x in line.split()])

            while True:
                line = fp.readline().strip()
                if not line:
                    break
                recList.append([float(x) for x in line.split()])

            while True:
                line = fp.readline().strip()
                if not line:
                    break
                motList.append([float(x) for x in line.split()])

            net.inputSynapses = numpy.array(senList)
            net.synapses = numpy.array(recList)
            net.outputSynapses = numpy.array(motList)


    def drawSynapseDebug(net):
        dbg = Application.debug
        dbg.clearstamps()

        screenSize = dbg.getscreen().screensize()
        dbg.goto(-screenSize[0] + 20, screenSize[1] - 20)

        sensorySyn = net.inputSynapses
        motorSyn = net.outputSynapses
        recurrentSyn = net.synapses

        x_start = dbg.xcor()

        for r in sensorySyn:
            dbg.goto(x_start, dbg.ycor() - 20)
            for e in r:
                cl = (float(e),float(e),float(e)) if e > 0 else (float(-e),0,0)
                dbg.color(cl)
                dbg.stamp()
                dbg.goto(dbg.xcor() + 20, dbg.ycor())

        dbg.goto(dbg.xcor() + 40, screenSize[1] - 20)

        x_start = dbg.xcor()

        for r in recurrentSyn:
            dbg.goto(x_start, dbg.ycor() - 20)
            for e in r:
                cl = (float(e),float(e),float(e)) if e > 0 else (float(-e),0,0)
                dbg.color(cl)
                dbg.stamp()
                dbg.goto(dbg.xcor() + 20, dbg.ycor())

        dbg.goto(dbg.xcor() + 40, screenSize[1] - 20)

        x_start = dbg.xcor()

        for r in motorSyn:
            dbg.goto(x_start, dbg.ycor() - 20)
            for e in r:
                e = e if e <= 1.0 else 1.0
                e = e if e >= -1.0 else -1.0
                cl = (float(e),float(e),float(e)) if e > 0 else (float(-e),0,0)
                dbg.color(cl)
                dbg.stamp()
                dbg.goto(dbg.xcor() + 20, dbg.ycor())


    def drawSpikeDebug(net):
        dbg = Application.spikes

        screenSize = dbg.getscreen().screensize()

        sensoryV = net.sensory.voltage
        motorV = net.motor.voltage
        recurrentV = net.voltage

        dbg.goto(-screenSize[0] + 20, -screenSize[1] + 70)

        x_start = dbg.xcor()

        for r,v in enumerate(sensoryV):
            nv = (v + 65) / 95
            nv = min(max(0.0, nv), 1.0)
            color = (float(nv), float(nv), float(nv))

            dbg.color(color)
            dbg.stamp()
            dbg.goto(dbg.xcor() + 20, dbg.ycor())

        dbg.goto(x_start, -screenSize[1] + 45)

        for r,v in enumerate(recurrentV):

            nv = (v + 65) / 95
            nv = min(max(0.0, nv), 1.0)
            color = (float(nv), float(nv), float(nv))

            dbg.color(color)
            dbg.stamp()
            dbg.goto(dbg.xcor() + 20, dbg.ycor())

        dbg.goto(x_start, -screenSize[1] + 20)

        for r,v in enumerate(motorV):
            nv = (v + 65) / 95
            nv = min(max(0.0, nv), 1.0)
            color = (float(nv), float(nv), float(nv))

            dbg.color(color)
            dbg.stamp()
            dbg.goto(dbg.xcor() + 20, dbg.ycor())


    def main():

        win = turtle.Screen()
        win.bgcolor('black')
        win.setup(width=800,height=600)
        win.tracer(0, 0)
        win.listen()

        Application.rootWindow = win.getcanvas().master

        agent = Agent()
        target = Target()
        controls = Text(200, 230, Application.controlText)

        Application.debug = Entity()
        Application.spikes = Entity()

        agent.setTarget(target)

        win.onkey(Application.registerClose, 'q')
        win.onkey(partial(Application.enableSynapseDebug, win, agent.net), 's')
        win.onkey(partial(Application.writeSynapses, agent.net), 'd')
        win.onkey(partial(Application.readSynapses, agent.net), 'l')
        win.onkey(partial(Application.setSpeed, 3), '3')
        win.onkey(partial(Application.setSpeed, 2), '2')
        win.onkey(partial(Application.setSpeed, 1), '1')
        win.onclick(target.goto)

        prev = time.clock_gettime(time.CLOCK_MONOTONIC)
        cur = prev

        debugAccum = 0.0
        updateAccum = 0.0
        agentRewardAccum = 0.0

        while Application.run:

            # Reset update accum when we change
            # the speed, so that we instantly
            # do the next update at the correct
            # time interval
            if Application.speedChanged is True:
                Application.speedChanged = False
                updateAccum = 0.0

            cur = time.clock_gettime(time.CLOCK_MONOTONIC)
            delta = cur - prev
            debugAccum += delta
            updateAccum += delta
            agentRewardAccum += delta

            Application.spikes.clearstamps()

            if Application.synapseDebug and debugAccum >= 0.5:
                Application.drawSynapseDebug(agent.net)
                debugAccum -= 0.5

            Application.drawSpikeDebug(agent.net)

            if agent.distance(target) < 20.0:
                target.onCollision()
                agent.reward(1 + 1 / agentRewardAccum)
                agentRewardAccum = 0.0

            if updateAccum >= Application.updateDelta:
                agent.update()
                updateAccum -= Application.updateDelta

            win.update()
            prev = cur

        turtle.bye()


if __name__ == '__main__':
    Application.main()
