from turtle import Screen
from turtle import Turtle

class Entity(Turtle):
    """
    Provides a base class inherited from Turtle for any entity that
    will exist during the course of the simulation
    """
    def __init__(self, pos=(0,0), color='red'):
        """
        Initializes the drawing settings for the Entity
        pos (tuple): The x and y coordinates of the entity
        color (str/tuple): The color of the entity for drawing
        """
        super().__init__()

        self.penup()
        self.shape('square')
        self.shapesize(1.5, 1.5)
        self.speed(0)

        if color:
            self.color(color)

        self.goto(pos)
        self.hideturtle()
        self.radians()

    def update(self):
        pass

