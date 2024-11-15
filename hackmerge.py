#!/usr/bin/env python3

import pickle

import network

net = None

id = -1
objs = []

with open('state', 'rb') as fp:
    try:
        while True:
            obj = pickle.load(fp)
            objs.append(obj)
            id += 1
            if obj.__class__.__name__ == 'Network':
                net = obj
    except:
        pass

if net != None:
    new_net = network.Network(net.num_neurons)
    new_net.w = net.w
    for k,v in net.params.__dict__.items():
        setattr(new_net.params, k, v)
    objs[id] = new_net
    with open('merged_state', 'wb') as fp:
        for obj in objs:
            pickle.dump(obj, fp)
