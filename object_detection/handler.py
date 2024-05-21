# diamorph_app is an application for detecting and classiying diatoms on images
# Copyright (C) 2022 Cyril Regan, Aishwarya Venkataramanan, Jeremy Fix, Martin Laviale, Cédric Pradalier

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Custom TorchServe model handler for YOLOv5 models.
"""
import io
import os
import base64
import time
import logging
from PIL import Image
import yaml
import requests
import numpy as np
import torch
import torchvision.transforms as tf
import cv2

from captum.attr import IntegratedGradients

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    rotate_non_max_suppression,
    rotate_scale_coords,
)
from utils.plots import (
    rotate_plot_crops,
)


logger = logging.getLogger(__name__)

# for GPU object detection inference
try:
    # if error in importing polygon_inter_union_cuda, polygon_b_inter_union_cuda,
    # please cd to ./iou_cuda and run "python setup.py install"
    from polygon_inter_union_cuda import (
        polygon_inter_union_cuda,
        polygon_b_inter_union_cuda,
    )

    polygon_inter_union_cuda_enable = True
    polygon_b_inter_union_cuda_enable = True

except Exception as e:
    print(
        f'Warning: "polygon_inter_union_cuda" and "polygon_b_inter_union_cuda" are not installed.'
    )
    print(f"The Exception is: {e}.")
    polygon_inter_union_cuda_enable = False
    polygon_b_inter_union_cuda_enable = False


def to_img_pil(img_open_cv):
    return Image.fromarray(cv2.cvtColor(img_open_cv, cv2.COLOR_BGR2RGB))


class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    """Image size (px). Images will be resized to this resolution before inference.
    """

    def __init__(self):
        # call superclass initializer
        super().__init__()
        logging.info(f"Torch version: {torch.__version__}")
        logging.info(f"Torch Cuda Available: {torch.cuda.is_available()}")

        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.profiler_args = {}

    def initialize(self, context):

        with open("detect_params.yaml") as f:
            detect_params = yaml.safe_load(f)  # load hyps
        if "imgsz" in detect_params:
            self.imgsz = detect_params["imgsz"]
        if "conf_thres" in detect_params:
            self.conf_thres = detect_params["conf_thres"]
        if "iou_thres" in detect_params:
            self.iou_thres = detect_params["iou_thres"]
        if "max_det" in detect_params:
            self.max_det = detect_params["max_det"]
        if "device" in detect_params:
            self.device = detect_params["device"]
        if "view_img" in detect_params:
            self.view_img = detect_params["view_img"]
        if "save_txt" in detect_params:
            self.save_txt = detect_params["save_txt"]
        if "save_conf" in detect_params:
            self.save_conf = detect_params["save_conf"]
        if "save_crop" in detect_params:
            self.save_crop = detect_params["save_crop"]
        if "nosave" in detect_params:
            self.nosave = detect_params["nosave"]
        if "classes" in detect_params:
            self.classes = detect_params["classes"]
        if "agnostic_nms" in detect_params:
            self.agnostic_nms = detect_params["agnostic_nms"]
        if "augment" in detect_params:
            self.augment = detect_params["augment"]
        if "update" in detect_params:
            self.update = detect_params["update"]
        if "project" in detect_params:
            self.project = detect_params["project"]
        if "name" in detect_params:
            self.name = detect_params["name"]
        if "exist_ok" in detect_params:
            self.exist_ok = detect_params["exist_ok"]
        if "line_thickness" in detect_params:
            self.line_thickness = detect_params["line_thickness"]
        if "hide_labels" in detect_params:
            self.hide_labels = detect_params["hide_labels"]
        if "hide_conf" in detect_params:
            self.hide_conf = detect_params["hide_conf"]
        if "half" in detect_params:
            self.half = detect_params["half"]
        if self.classes == "":
            self.classes = None

        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None

        self.manifest = context.manifest

        model_dir = properties.get("model_dir")

        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")
        if model_file:
            logger.debug("Loading eager model")
            self.model = self._load_pickled_model(model_dir, model_file, model_pt_path)
            self.model.to(self.device)
        else:
            logger.debug("Loading torchscript model")
            if not os.path.isfile(model_pt_path):
                raise RuntimeError("Missing the model.pt file")

        self.model = attempt_load(
            model_pt_path, map_location=self.device
        )  # load FP32 model
        stride = int(self.model.stride.max())  # model stride+

        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )  # get class names

        self.model.eval()
        print("Model file %s loaded successfully", model_pt_path)

    def preprocess(self, data):
        """Converts input images to float tensors.
        Args:
            data (List): Input data from the request in the form of a list of image tensors.
        Returns:
            Tensor: single Tensor of shape [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
        """
        images = []

        transform = tf.ToTensor()

        # load images
        # taken from https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py
        images_brg = []
        # handle if images are given in base64, etc.
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)
            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images_brg.append(np.array(image))

            # force convert to tensor
            # and resize to [img_size, img_size]
            image = transform(image)

            images.append(image)

        # convert list of equal-size tensors to single stacked tensor
        # has shape BATCH_SIZE x 3 x IMG_SIZE x IMG_SIZE
        images_tensor = torch.stack(images).to(self.device)

        # Check the number of channels and correct it if necessary
        # Although the images might be considered as colored images
        # These are indeed B&W images. Although these are B&W
        # the model expects input tensors with 3 channels
        _, cin, _, _ = images_tensor.shape
        logging.info(f"Output of preprocess is a tensor of shape {images_tensor.shape}")

        return images_tensor, images_brg

    def postprocess(self, inference_output, data_rgb):

        # perform NMS (nonmax suppression) on model outputs

        pred = rotate_non_max_suppression(
            inference_output[0],
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )

        # initialize empty list of detections for each image
        detections = [[] for _ in range(len(pred))]

        for i, det in enumerate(pred):  # axis 0: for each image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = rotate_scale_coords(
                    data_rgb[i].shape[:2], det[:, :4], data_rgb[i].shape
                ).round()
                it_obj = 0
                for *xywh, real, imagin, conf, cls in reversed(det):
                    # index of predicted class
                    class_idx = int(cls)
                    # get label of predicted class
                    # if missing, then just return class idx
                    # self.mapping.get(str(class_idx), class_idx)
                    label = self.names[class_idx]

                    crop = rotate_plot_crops(xywh, real, imagin, data_rgb[i], it_obj)

                    crop_img_new = crop["crop_img_new"]

                    # cv2.imwrite("crop_img_new_{}.png".format(
                    #     it_obj), crop_img_new)

                    img = to_img_pil(crop_img_new)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format="PNG")
                    byte_im = img_byte_arr.getvalue()

                    class_port = os.environ["PORT_CLASS_INF"]

                    try:
                        response = requests.post(
                            "http://classif:" + class_port + "/predictions/Mobilenet",
                            data=byte_im,
                        ).json()  # github api

                    except requests.exceptions.HTTPError as e:
                        # Whoops it wasn't a 200
                        return "Error: " + str(e)

                    it_obj += 1

                    if (
                        list(response.keys())[0] == "code"
                        or list(response.values())[0] == 503
                    ):
                        class_classif = " "
                        class_classif_conf = 0.0
                    else:
                        class_classif = list(response.keys())[0]
                        class_classif_conf = list(response.values())[0]

                    detections[i].append(
                        {
                            "x": xywh[0].item(),
                            "y": xywh[1].item(),
                            "w": xywh[2].item(),
                            "h": xywh[3].item(),
                            "real": real.item(),
                            "imagin": imagin.item(),
                            "OD confidence": conf.item(),
                            "class": class_classif,
                            "class confidence": class_classif_conf,
                        }
                    )

        # format each detection
        return detections

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context

        start_time = time.time()

        self.context = context

        metrics = self.context.metrics

        data_preprocess, data_rgb = self.preprocess(data)

        logging.info(
            f"For inference, preprocessed data has shape {data_preprocess.shape}; data_rgb has shape {data_rgb[0].shape}"
        )
        if not self._is_explain():
            inference_output = self.model(data_preprocess)

            output = self.postprocess(inference_output, data_rgb)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )

        return output

    def _is_explain(self):
        return False
