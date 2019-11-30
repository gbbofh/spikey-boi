# Spikey-Boi

### A moving triangle that is supposed to learn how to find a target and does, kind of.

## What it is

This is an extension of another repository of mine, spiking-python, to add
support for spike-timing dependent plasticity to the neural network model. The
implementation of STDP is relatively simple, and currently has some efficiency
issues that need worked out in that particular region of code.

It has been modified here to control a virtual agent, modelled after a simple
robot.

### The Agent

The agent here is represented by a red triangle. It has three sensory inputs,
and two motor outputs (for the time being). The inputs to the network are a
function of the relative angle between the front of the agent, and the target
(green circle). Which input responds, as well as how it responds, depends upon
the angle between the agent and the target. In order to attempt to drive the
agent to seek the target, the sensory input is strongest at the front, and
decays along either side.

The motor output of the agent is produced by sampling the spike frequency of
each motor neuron over a period of time. This period of time is a defined
constant in the Agent class, and is currently set to 100 timesteps.

### The Display

This program is capable of displaying some helpful information about the agent.

In the top left, the synaptic matrix is displayed. Weights can be any value
in the range [-1.0, 1.0]. Negative weights are displayed as red, while positive
weights are displayed as white.

Below this, the current membrane potential of each neuron is displayed.
The current manner of how these are displayed is somewhat clunky, and I intend
to clean this up in the future.

Controls are displayed in the upper right.

### The Network

The spiking network is built upon Eugene M. Izhikevich's model, published in
2003, with the addition of a parameter for each neuron to facilitate STDP.

Initial implementations featured a single recurrent spiking network, however
this proved problematic for several reasons; the most notable being that motor
neurons would form synaptic connections with one another (as well as the input
neurons), which would directly interfere with one another.

The solution to this problem was to separate the network into three segments (or
layers). In this manner, we actually have three synaptic matrices: One for
synapses from the input neurons to the recurrent layer; one for the recurrent
layer itself; and one for neurons from the recurrent layer to the output layer.

In this manner we can form connections from the input neurons to any neurons in
the recurrent layer, and we can form connections from any neurons in the
recurrent layer to any neurons in the output layer.

In addition to STDP, a positive reward is given to the agent for reaching the
target -- in the future a punishment will also be given to the agent if it takes
too long to reach the target. I did not have time to implement this before it
was presented for class.

## Requirements

 * Python (Version >= 3.1)
 * SciPy
 * NumPy

## Usage

```bash
python3 main.py
```

Right now the program is relatively minimal, but more features are going to be
coming in the near future. One of the most important will be modifying the
save/load functionality to include information about the entire network -- not
just synaptic connections, but also parameters for the spiking model.

## To Do

There are a lot of optimizations that I wish to do on this code. Noteably, I
wish to add a live graphical display of spikes via matplotlib, so that the
history of spikes can be observed. This is something that I have available in
another repository but integrating it here has proven to be somewhat
challenging. Nevertheless, implementing it would be incredibly helpful, so that
is a major goal in the near future.

