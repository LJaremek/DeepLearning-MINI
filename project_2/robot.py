from turtle import Turtle, Screen


class Robot:
    def __init__(self) -> None:
        self.screen = Screen()
        self.screen.title = "voice control Robot"
        self.screen.window_height = 400
        self.screen.window_width = 400

        self.t = Turtle()
        self.t.shape("circle")
        self.t.left(90)

        self.text_t = Turtle()
        self.text_t.hideturtle()
        self.text_t.penup()

    def display_text(self, text: str, x: int, y: int) -> None:
        self.text_t.clear()
        self.text_t.goto(x, y)
        self.text_t.write(text, align="left", font=("Arial", 14, "normal"))

    def up(self) -> None:
        self.t.forward(100)

    def left(self) -> None:
        self.t.left(90)
        self.t.forward(100)
        self.t.right(90)

    def right(self) -> None:
        self.t.right(90)
        self.t.forward(100)
        self.t.left(90)

    def down(self) -> None:
        self.t.backward(100)

    def start(self) -> None:
        """
        NOTE:
            Run this function after voice recognition thred is started!
            Please check `robot_example_usage.py`.
        """
        self.screen.mainloop()
