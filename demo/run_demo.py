import os
import threading
import time
import cv2

import asyncio
import subprocess
import numpy as np
import multiprocessing as mp

STATE = 0

def process_thread():
    t = threading.Thread(target=_process_thread)
    t.daemon = True
    t.start()


def _process_thread():
    global STATE
    if os.path.exists("output"):
        subprocess.run(["rm", "-rf", "output"])
    os.makedirs("output")
    asyncio.run(inference_subprocess())
    STATE = 1


async def inference_subprocess():
    processes = []

    input_video_path = "./input_video"
    output_path = "./output"
    input_video_names = os.listdir(input_video_path)

    input_paths = [
        os.path.join(input_video_path, inpu_video_name)
        for inpu_video_name in input_video_names
    ]
    output_paths = [
        os.path.join(output_path, inpu_video_name.split(".")[0])
        for inpu_video_name in input_video_names
    ]

    # If we have 2 NPUs with 2 PEs each, we can spawn 4 subprocesses by utilizing a single PE at a time.
    for idx in range(4):
        if os.path.exists(output_paths[idx]):
            subprocess.run(["rm", "-rf", output_paths[idx]])
        os.mkdir(output_paths[idx])

        processes.append(
            await asyncio.create_subprocess_exec(
                "python",
                "demo_subprocess.py",
                input_paths[idx],
                output_paths[idx],
                "warboy(1)*1",  # Use one single-PE warboy (or, "warboy*1")
            )
        )
    await asyncio.gather(*(process.wait() for process in processes))


if __name__ == "__main__":
    process_thread()
    # streaming or other..
    try:
        while STATE == 0:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    print("Exit!!")
