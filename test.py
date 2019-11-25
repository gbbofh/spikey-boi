import network as net

import scipy.stats as stats

import time

n = net.Network(15, 1, net.Layer(3, 0), net.Layer(2, 0))

while True:
    #n.update(5 * stats.norm.rvs(size=3))
    n.update([10.0, 10.0, 10.0])
    print(n.sensory.pSpikes, n.pSpikes, n.motor.pSpikes)
    time.sleep(0.5)

