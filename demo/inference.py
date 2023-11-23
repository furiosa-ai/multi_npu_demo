import os
import asyncio
import typer
import time

import cv2
import numpy as np
import torch

from utils.preprocess import *
from utils.postprocess import *
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def run_inference(input_dataQ, device):
    rt_config = FuriosaRTModelConfig(
        name="detection", npu_device=device, model="yolov7.enf", worker_num=2
    )
    detection_model = FuriosaRTModel(rt_config)
    asyncio.run(run(detection_model, input_dataQ))


async def run(detection_model, input_dataQ):
    await detection_model.load()
    t1 = time.perf_counter()
    await asyncio.gather(
        *(task(detection_model, input_dataQ, worker_id) for worker_id in range(2))
    )
    t2 = time.perf_counter()
    print(t2 - t1)


async def task(detection_model, input_dataQ, worker_id):
    while True:
        if input_dataQ.empty():
            await asyncio.sleep(0.001)
        else:
            input_img, img_idx, output_path = input_dataQ.get()
            if img_idx < 0:
                break

            if img_idx % 2 == worker_id:
                await process(detection_model, input_img, img_idx, output_path)


async def process(detection_model, input_img, img_idx, output_path):
    output_img_path = os.path.join(output_path, "%010d.png" % img_idx)

    input_, preproc_params = preproc(input_img)
    output = await predict(detection_model, input_)
    predictions = postproc(output, 0.65, 0.35)

    assert len(predictions) == 1, f"{len(predictions)=}"

    predictions = predictions[0]

    num_predictions = predictions.shape[0]
    if num_predictions == 0:
        cv2.imwrite(output_img_path, input_img)
        return

    bboxed_img = draw_bbox(input_img, predictions, preproc_params)
    cv2.imwrite(output_img_path, bboxed_img)


async def predict(detection_model, input_tensor):
    output = await detection_model.run([input_tensor])
    return output
