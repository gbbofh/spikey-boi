from entity import Entity


class Text(Entity):

    def __init__(self, x, y=None, text=''):
        super().__init__()

        self.color('white')
        self.goto(x, y)
        self.write(text, font=('Arial', 11, 'normal'))
        self.hideturtle()

