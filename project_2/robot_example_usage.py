import threading
import random
import time

from robot import Robot


class VoiceCommandsMock:
    def __init__(self, moves: list[str] = None) -> None:
        if moves is None:
            self.moves = ["up", "down", "left", "right", "silent"]
        else:
            self.moves = moves

    def detect_move(self, voice=None) -> str:
        return random.choice(self.moves)


def voice_commands_recognition(robot: Robot) -> None:
    voice_recognizer = VoiceCommandsMock()

    while True:
        new_command = voice_recognizer.detect_move()
        print(new_command)

        robot.display_text(new_command, -300, 220)

        if new_command == "silent":
            pass
        if new_command == "up":
            robot.up()
        if new_command == "down":
            robot.down()
        if new_command == "left":
            robot.left()
        if new_command == "right":
            robot.right()

        time.sleep(2)


if __name__ == "__main__":
    my_robot = Robot()

    thread = threading.Thread(
        target=voice_commands_recognition,
        args=(my_robot,)
        )

    thread.daemon = True
    thread.start()

    my_robot.start()
