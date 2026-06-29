import winsound


def alert():
    frequency = 2500
    duration = 1000

    winsound.Beep(
        frequency,
        duration
    )