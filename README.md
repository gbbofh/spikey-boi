# Description
This project is a rewrite of the original "spikey-boi" project, of which I was
one of two authors while taking a Computational Neuroscience class during my undergraduate degree.

The codebase has been completely rewritten from scratch, and has been ported from
using python turtle, to using pygame for rendering. The core idea is still the same
but a number of performance improvements have been made, in addition to providing
a more realistic and functional model of reward-modulated STDP.

The simulation will perform a periodic auto-save of training progress to a file
named `autosave`. Automatic loading from this file is not currently implemented.

The state can also be saved on-demand to the `state` file. I have included a
pre-trained model which has been training for a total of ~50 hours already. To
load this model, you can press `backspace` once the simulation starts.

## Implementation Details
For simplicity, I replaced the Izhekivich model neurons with a Leaky-Integrate and Fire model.
There are a total of 15 input neurons, defined in the `Agent` class, and 2 motor neurons.

The input neurons respond to the following environmental features:

| id  | environmental feature                                                      |
| --- | -------------------------------------------------------------------------- |
| 0   | Signed distance along the x-axis from the agent to the target              |
| 1   | Signed distance along the y-axis from the agent to the target              |
| 2   | Negative signed distance along the x-axis from the agent to the target     |
| 3   | Negative signed distance along the y-axis from the agent to the target     |
| 4   | Euclidean distance from the agent to the target                            |
| 5   | Distance from agent to wall nearby wall (if wall is to the left)           |
| 6   | Distance from agent to wall nearby wall (if wall is to the right)          |
| 7   | Agent has collected the target recently                                    |
| 8   | Angle of the agent to the target $\left[\frac{-pi}{4},\frac{pi}{4}\right]$ |
| 9   | Agent has moved closer to the target this frame                            |
| 10  | Cosine of the angle between the agent and target                           |
| 11  | Sine of the angle between the agent and target                             |
| 12  | Signed position relative to the center of the display (x-axis)             |
| 13  | Signed position relative to the center of the display (y-axis)             |
| 14  | Negative signed position relative to the center of the display (x-axis)    |
| 15  | Negative signed position relative to the center of the display (y-axis)    |

Most of these sensory inputs decay exponentially, or are passed through an
exponential function before being given to the input neuron as an injected current.

The firing rate of all neurons is calculated every update, according to the following
formula:
```math

Fr_{i} = \sum_{k=1}^{1000} \frac{S_{i,k}}{k} \cdot \frac{1000}{\tau} \\

```

Each spike is weighted according to how many simulation steps have elapsed since it occurred.

This firing rate is used to drive the agent to turn and move in the direction it is currently facing,
depending on whether one or both of the motor neurons is active.

Learning is done once every simulation timestep, and incorporates not only spikes, but also
sub-threshold membrane potentials, in addition to reward eligibility.

Unlike the original model, which used 2 feed-forward layers and a recurrent hidden layer, this
model uses only a single fully recurrent layer. Any structure is entirely decided during the training process.

# Usage
To run, simply execute `main.py`. This will open the graphical display and start the simulation.
The simulation should update every 4 ms, and the actual neural network is updated using a time delta of 0.1 ms.
This means that the simulation runs 25 times slower than real-time, in ideal conditions.


## Keybindings

I have added a number of keybindings to provide debugging / visualization information at runtime.

| key       | description                                         |
| --------- | --------------------------------------------------- |
| ~         | Enable debugging                                    |
| 1         | Show / hide membrane potentials                     |
| 2         | Show / hide firing rates                            |
| 3         | Show / hide membrane potential correlation matrix   |
| 4         | Show / hide spikes                                  |
| 5         | Show / hide synaptic weights                        |
| 6         | Show / hide reward modulation                       |
| 7         | Show / hide reward eligibility                      |
| 8         | Show / hide STDP update heatmap                     |
| 9         | Show / hide input current                           |
| 0         | Enable / disable convolution kernels (blurring)     |
| /         | Show / hide position trace                          |
| n         | Enable / disable gaussian noise input               |
| i         | Enable / disable sensory input                      |
| q         | Enable / disable dynamic scaling for gaussian noise |
| g         | Enable / disable 3D connectome                      |
| space     | Save the current simulation state to `state` file   |
| backspace | Load the simulation state from `state` file         |

# Future Work

I plan on continuing to work on this project in the future. I would like to
eventually implement some more visualizations to show network architecture
and activity during the learning process. One visualization I am currently exploring
is an interactive 3D view of the network, where the sensory input and motor neurons
are displayed separately from the rest of the network. This display does not currently
work in real-time, as it relies on plotly. I may eventually look into efficiently
implementing such a plot using pygame, or I may instead migrate this entire application
over to another language like C#.

One thing I would like to explore is allowing for populations of agents to exist at once,
possibly with evolutionary / genetic algorithms in addition to learning to drive some features
like network size and allowing some structure to become 'locked into place,' so to speak; however
doing this would require a considerable amount of work, and most likely would necessitate switching
to another language and / or a dedicated game engine for efficiency reasons.
