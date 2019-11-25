from entity import Entity

class Target(Entity):

    def __init__(self):
        super().__init__()

        self.shape('circle')
        self.color('green')
        self.showturtle()


    def update(self):
        pass


    def on_collision(self):
        pos = numpy.random.uniform(0.0, 1.0, size=2)
        pos *= self.getscreen().screensize()
        self.goto(pos)

