from entity import Entity


class Text(Entity):

    FONT = ('Arial', 11, 'normal')

    """
    Specialized turtle-based entity used for drawing text at a position
    """
    def __init__(self, x, y=None, text=''):
        """
        Constructs a turtle intended for drawing text at a position
        Parameters:
        x (int): The x-position of the turtle, or a tuple of x,y coordinates
        y (int): The y-position of the turtle, or None
        text (str): The string to render
        """
        super().__init__()

        self.color('white')
        self.goto(x, y)
        self.write(text, font=Text.FONT)
        self.hideturtle()


    def setText(self,text=''):
        self.clear()
        self.write(text, font=Text.FONT)

