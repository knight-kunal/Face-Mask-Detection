import csv
import os
from datetime import datetime


def save_log(total, mask, no_mask):

    if not os.path.exists("logs"):
        os.makedirs("logs")


    file_path = "logs/detection_history.csv"


    file_exists = os.path.isfile(file_path)


    with open(file_path, "a", newline="") as file:

        writer = csv.writer(file)


        if not file_exists:
            writer.writerow(
                [
                    "Time",
                    "Total People",
                    "Mask",
                    "No Mask"
                ]
            )


        writer.writerow(
            [
                datetime.now().strftime("%H:%M:%S"),
                total,
                mask,
                no_mask
            ]
        )