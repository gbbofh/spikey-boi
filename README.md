# Spikey-Boi

### A moving triangle that is supposed to learn how to find a target and does, kind of.

## What it is

This is an extension of another repository of mine, spiking-python, to add
support for spike-timing dependent plasticity to the neural network model. The
implementation of STDP is relatively simple, and currently has some efficiency
issues that need worked out in that particular region of code.

This is *not* intended to be a fully functional implementation, and it is still
under construction ( and will be for the indefinite future, as this is a subject
that interests me greatly ). This started out as a project for an undergraduate
Computer Science class, which introduced the foundational topics of
computational neuroscience.


## Requirements

 * SciPy
 * NumPy

SciPy and NumPy are used for their extremely fast array indexing and creation,
as well as for the generation of arrays of random numbers. In addition to this,
the Turtle library is used to display the agent, as well as its neuronal
activity and (optionally, if you uncomment the line to enable it) synaptic
weights. *WARNING*, drawing of synaptic weights is very time consuming, and will
cause the application to run significantly slower.

## To Do

The agent class and the associated main function is an absolute mess -- I am
well aware of this. In the future they will be separated into two different
files, and the target code will be pulled out of agent and moved to its own
class. This was done as a hasty prototype for my groups project, just to see if
it would work. Time devoted to this entire prototype was ~1 week of development,
whilst simultaneously writing a report on its development.

