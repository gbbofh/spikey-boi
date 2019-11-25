from turtle import Screen
from turtle import Turtle

class Entity(Turtle):
    def __init__(self, pos=(0,0), color='red'):
        super().__init__()

        self.penup()
        self.shape('square')
        self.speed(0)

        if color:
            self.color(color)

        self.goto(pos)
        self.hideturtle()
        self.radians()

    def update(self):
        pass

