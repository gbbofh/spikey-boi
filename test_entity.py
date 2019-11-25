import time
import turtle

from functools import partial

from text import Text
from target import Target
from entity import Entity
from new_agent import Agent


class Application():

    run = True
    synapseDebug = False
    controlText = '\n'.join(('Press Q to Exit',
                            'Press S to Draw Synapses',
                            'Press D to Dump Synapses to File',
                            'Press L to Load Synapses from File'))
    debug = None

    def registerClose():
        Application.run = False


    def enableSynapseDebug(win, net):
        Application.synapseDebug = not Application.synapseDebug
        if not Application.debug:
            Application.debug = Entity()
        if not Application.synapseDebug:
            Application.debug.clearstamps()


    def writeSynapses(net):
        sensorySyn = net.inputSynapses
        motorSyn= net.outputSynapses
        recurrentSyn = net.synapses

        with open('agent.syn', 'w') as fp:
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
                cl = (float(e),float(e),float(e)) if e > 0 else (float(-e),0,0)
                dbg.color(cl)
                dbg.stamp()
                dbg.goto(dbg.xcor() + 20, dbg.ycor())


    def main():

        win = turtle.Screen()
        win.bgcolor('black')
        win.setup(width=800,height=600)
        win.tracer(0, 0)
        win.listen()

        agent = Agent()
        target = Target()
        controls = Text(200, 240, Application.controlText)

        win.onkey(Application.registerClose, 'q')
        win.onkey(partial(Application.enableSynapseDebug, win, agent.net), 's')
        win.onkey(partial(Application.writeSynapses, agent.net), 'd')
        win.onkey(lambda: None, 'l')

        while Application.run:

            # TODO: Run this only every few hundred ms so that it isn't
            # eating up so much time.
            # turtles ontimer method *will not* work, because it
            # does not play nice with functools.partial
            if Application.synapseDebug:
                Application.drawSynapseDebug(agent.net)

            agent.update()
            win.update()

        turtle.bye()


if __name__ == '__main__':
    Application.main()
