import numpy

from entity import Entity

class Target(Entity):

    def __init__(self):
        super().__init__()

        self.shape('circle')
        self.color('green')
        self.showturtle()
        self.onCollision()


    def update(self):
        pass


    def onCollision(self):
        pos = numpy.random.uniform(-1.0, 1.0, size=2)
        pos *= self.getscreen().screensize()
        self.goto(pos)
