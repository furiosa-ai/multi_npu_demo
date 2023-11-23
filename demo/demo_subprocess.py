import os
import threading
import time
import cv2
import typer
import asyncio
import subprocess
import numpy as np
import multiprocessing as mp
from inference import run_inference

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def demo_process(input_paths, output_path, device):
    inference_dataQ = mp.Queue()
    d_proc = mp.Process(target=run_inference, args=(inference_dataQ, device))
    d_proc.start()

    img_id = 0
    cap = cv2.VideoCapture(input_paths)
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        inference_dataQ.put((frame, img_id, output_path))
        img_id += 1
    cap.release()

    for i in range(2):
        inference_dataQ.put((None, -1, None))

    d_proc.join()
    inference_dataQ.close()
    return


if __name__ == "__main__":
    app()
