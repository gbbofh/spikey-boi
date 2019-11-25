import network

import argparse
import datetime
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plot
import matplotlib.animation as animation

#k = 100
timeScale = 500
#net = network.Network(7 * k, 3 * k)
inputLayer = network.Layer(50, 25)
outputLayer = network.Layer(50, 25)
net = network.Network(380, 22, inputLayer, outputLayer)
#net = network.Network(8 * k, 2 * k)

fig = plot.figure(figsize=(6,6))

ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlim(0, timeScale)
ax1.set_ylim(-1, net.totalNum + 1)

xl, xr = ax1.get_xlim()
yl, yh = ax1.get_ylim()

ax1.set_aspect(abs((xr - xl) / (yl - yh)))
ax1.axhline(color='r', y=net.numEx - 0.5, xmax=xr)
vline = ax1.axvline(color='k', x=1, ymin=yl, ymax=yh)

trans = ax1.get_xaxis_transform()
ax1.annotate('inh.',
             xy=(timeScale + 1, (2 * net.numEx + net.numIn) / (2 * net.totalNum)),
             xycoords=trans)

ax1.annotate('exc.',
             xy=(timeScale + 1, net.numEx / (2 * net.totalNum)), xycoords=trans)

line, = ax1.plot([],[], ',k')

data = [[] for x in range(timeScale)]

def init():
    line.set_data([],[])
    return line, vline

def animate(i):
    inputs = np.ones(net.sensory.totalNum)
    inputs[0 : net.sensory.numEx] = 5.0 * stat.norm.rvs(size=net.sensory.numEx)
    inputs[net.sensory.numEx : ] = 2.0 * stat.norm.rvs(size=net.sensory.numIn)
    spikes = net.update(inputs)
    #net.input[0 : net.numEx] = 5.0 * stat.norm.rvs(size=net.numEx)
    #net.input[net.numEx : ] = 2.0 * stat.norm.rvs(size=net.numIn)
    #spikes = net.update()
    data[i % timeScale] = spikes
    times = []
    spikes = []
    for k, d in enumerate(data):
        times += [k for s in d]
        spikes += [s for s in d]
    line.set_data(times, spikes)
    #ax1.plot([i % 100 for x in spikes], spikes, ',k')
    vline.set_xdata(((i + 1) % timeScale, (i + 1) % timeScale))

    return line, vline


# For some reason interval < 110 on my machine
# causes a KeyError in Tkinter. But the graph still
# outputs, so I don't really know what to do here.
# Any suggestions are welcome...
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               interval=60, blit=True)
plot.show()

