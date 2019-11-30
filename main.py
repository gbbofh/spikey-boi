import os
import time
import turtle

import tkinter
import tkinter.filedialog as filedialog

from functools import partial

import numpy

from text import Text
from target import Target
from entity import Entity
from agent import Agent


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


    def enableNoisy(net):
        net.noisy = not net.noisy
        net.sensory.noisy = not net.sensory.noisy
        net.motor.noisy = not net.motor.noisy


    def enableSynapseDebug():
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


    # TODO:
    # Have these functions serialize everything there is to know
    # about a network (or agent?) so that they can be completely
    # loaded. This should be done because neuron parameters
    # vary for every network generated.
    def writeSynapses(agent):
        types = [('all files', '.*'), ('synapses', '.syn')]

        net = agent.net

        sensorySyn = net.inputSynapses
        recurrentSyn = net.synapses
        motorSyn = net.outputSynapses

        ans = filedialog.asksaveasfilename(parent=Application.rootWindow,
                                                    initialdir=os.getcwd(),
                                                    title='Save Synapses',
                                                    filetypes=types)
        if not ans:
            return

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


    # TODO:
    # Have these functions serialize everything there is to know
    # about a network (or agent?) so that they can be completely
    # loaded. This should be done because neuron parameters
    # vary for every network generated.
    def readSynapses(agent):
        types = [('all files', '.*'), ('synapses', '.syn')]
        net = agent.net

        ans = filedialog.askopenfilename(parent=Application.rootWindow,
                                                    initialdir=os.getcwd(),
                                                    title='Load Synapses',
                                                    filetypes=types)
        if not ans:
            return

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

            inputSyn = numpy.array(senList)
            recurrentSyn = numpy.array(recList)
            motorSyn = numpy.array(motList)

            if (inputSyn.shape != net.inputSynapses.shape or
                recurrentSyn.shape != net.synapses.shape or
                motorSyn.shape != net.synapses.shape):
                print('Unable to transfer weights. Size mismatch.')

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
                e = e if e < 1.0 else 1.0
                e = e if e > -1.0 else -1.0
                cl = (float(e),float(e),float(e)) if e > 0 else (float(-e),0,0)
                dbg.color(cl)
                dbg.stamp()
                dbg.goto(dbg.xcor() + 20, dbg.ycor())

        dbg.goto(dbg.xcor() + 40, screenSize[1] - 20)

        x_start = dbg.xcor()

        for r in recurrentSyn:
            dbg.goto(x_start, dbg.ycor() - 20)
            for e in r:
                e = e if e < 1.0 else 1.0
                e = e if e > -1.0 else -1.0
                cl = (float(e),float(e),float(e)) if e > 0 else (float(-e),0,0)
                dbg.color(cl)
                dbg.stamp()
                dbg.goto(dbg.xcor() + 20, dbg.ycor())

        dbg.goto(dbg.xcor() + 40, screenSize[1] - 20)

        x_start = dbg.xcor()

        for r in motorSyn:
            dbg.goto(x_start, dbg.ycor() - 20)
            for e in r:
                e = e if e < 1.0 else 1.0
                e = e if e > -1.0 else -1.0
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

        # TODO:
        # Lets add a population of agents, initialized to
        # random positions, with their own network.
        # Maybe we can see if competition changes anything
        # Also consider adding a pool of targets
        # and give each agent the information about the
        # closest available target.
        agent = Agent(21, 9)
        target = Target()

        Application.debug = Entity()
        Application.spikes = Entity()

        controls = Text(200, 230, Application.controlText)
        motorDisplay = Text(200, 200)

        Application.debug.shapesize(1.0, 1.0)
        Application.spikes.shapesize(1.0, 1.0)

        agent.setTarget(target)

        win.onkey(Application.registerClose, 'q')
        win.onkey(Application.enableSynapseDebug, 's')
        win.onkey(partial(Application.writeSynapses, agent), 'd')
        win.onkey(partial(Application.readSynapses, agent), 'l')
        win.onkey(partial(Application.setSpeed, 3), '3')
        win.onkey(partial(Application.setSpeed, 2), '2')
        win.onkey(partial(Application.setSpeed, 1), '1')
        win.onkey(partial(Application.enableNoisy, agent.net), 'n')
        win.onclick(target.goto)

        prev = time.clock_gettime(time.CLOCK_MONOTONIC)
        cur = prev

        debugAccum = 0.0
        updateAccum = 0.0

        # NOTE: We want this to be separate from realtime
        # because the agent should still get a reward if we
        # change the speed setting.
        agentRewardAccum = 0

        foundCount = 0

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


            mdt = 'MF: {} TC: {}'.format(agent.motorFrequency, foundCount)
            motorDisplay.setText(mdt)

            Application.spikes.clearstamps()

            if Application.synapseDebug and debugAccum >= 0.5:
                Application.drawSynapseDebug(agent.net)
                debugAccum -= 0.5

            Application.drawSpikeDebug(agent.net)

            if agent.distance(target) < 30.0 or agentRewardAccum > 250:
                target.onCollision()
                foundCount += 1
                agent.reward((1 + 1 / agentRewardAccum))
                agentRewardAccum = 0
                agent.clear()


            screenSize = agent.getscreen().screensize()
            agentX = agent.xcor()
            agentY = agent.ycor()
            agentX = agentX if agentX < screenSize[0] - 30 else -agentX + 30
            agentX = agentX if agentX > -screenSize[0] + 30 else -agentX - 30
            agentY = agentY if agentY < screenSize[1] - 30 else -agentY + 30
            agentY = agentY if agentY > -screenSize[1] + 30 else -agentY - 30
            agent.penup()
            agent.goto(agentX, agentY)
            agent.pendown()


            if updateAccum >= Application.updateDelta:
                agent.update()
                updateAccum -= Application.updateDelta
                agentRewardAccum += 0.0005 # Update the delta every logical timestep

            win.update()
            prev = cur

        turtle.bye()


if __name__ == '__main__':
    Application.main()
