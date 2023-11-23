import os
import threading
import time
import cv2

import asyncio
import subprocess
import numpy as np
import multiprocessing as mp

# from fastapi import FastAPI, Response
# from fastapi.responses import StreamingResponse, FileResponse


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

    device = ["npu0pe0"]
    input_paths = ["/home/furiosa/hackaton_demo/demo/input_video/road_trafifc.mp4"]
    output_paths = ["/home/furiosa/hackaton_demo/demo/output"]
    for idx in range(len(device)):
        processes.append(
            await asyncio.create_subprocess_exec(
                "python",
                "demo_subprocess.py",
                input_paths[idx],
                output_paths[idx][0],
                device[idx],
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
