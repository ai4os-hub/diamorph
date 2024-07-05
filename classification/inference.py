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

"""
ModelHandler defines a custom model handler.
"""

import cv2
import numpy as np
import math
import time
import itertools
from PIL import Image
import base64
import os
import io
from ts.context import Context
from captum.attr import IntegratedGradients
from torchvision import transforms
import torch.nn.functional as F
import torch

from ts.utils.util import load_label_mapping
from abc import ABC
import logging
from ete3 import Tree

# import data_utilslog/mobilenet_223/
logger = logging.getLogger(__name__)


"""
Util files for TorchServe
"""

logger = logging.getLogger(__name__)

ipex_enabled = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
    try:
        import intel_extension_for_pytorch as ipex

        ipex_enabled = True
    except ImportError as error:
        logger.warning(
            "IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX."
        )


def convert_to_square(image, new_size=None, padding=1):
    """_summary_

    Args:
        image (_type_): _description_
        new_size (_type_, optional): _description_. Defaults to None.
        padding (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # kernel = np.ones((3, 3), np.uint8)
    # image_b = cv2.dilate(image, kernel, iterations = 2)
    crop = 1
    image = image[crop:-crop, crop:-crop]
    # Preprocessing
    if not new_size is None:
        ratio = new_size / np.max(image.shape)
        image = cv2.resize(
            image,
            dsize=(
                math.floor(ratio * image.shape[1]) - 2 * padding,
                math.floor(ratio * image.shape[0]) - 2 * padding,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    # Converting to square
    square_size = np.max(image.shape)
    h, w = image.shape[0], image.shape[1]
    delta_w, delta_h = square_size - w, square_size - h
    left, top = delta_w // 2, delta_h // 2
    blur_size = int(np.max(image.shape) / 4) * 2 + 1
    blured_image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    square_image_blurred = cv2.copyMakeBorder(
        blured_image,
        top + padding,
        delta_h - top + padding,
        left + padding,
        delta_w - left + padding,
        cv2.BORDER_REPLICATE,
    )
    square_image = square_image_blurred.copy()
    square_image[
        top + padding : top + h + padding,
        left + padding : left + w + padding,
    ] = image.copy()

    # Seamless cloning
    height = square_image_blurred.shape[0]
    width = square_image_blurred.shape[1]
    mask_ref = np.zeros_like(square_image).astype("uint8")
    mask_ref[top + 1 : top + h + 1, left + 1 : left + w + 1] = 255
    center = (height // 2, width // 2)
    src = expand(square_image)
    dst = expand(square_image_blurred)
    final_image = cv2.seamlessClone(
        src, dst, mask_ref, center, cv2.NORMAL_CLONE
    )

    if not new_size is None:
        final_image = cv2.resize(
            final_image,
            dsize=(new_size, new_size),
            interpolation=cv2.INTER_CUBIC,
        )
    return final_image


def expand(image):
    if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image


def load_checkpoint(filepath, device):

    checkpoint = torch.load(filepath)
    print("checkpoint", checkpoint)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(device)

    model.eval()
    return model


def map_class_to_label(probs, mapping=None, lbl_classes=None):
    """
    Given a list of classes & probabilities, return a dictionary of
    { friendly class name -> probability }
    """
    if not (isinstance(probs, list) and isinstance(probs, list)):
        raise Exception(
            "Convert classes to list before doing mapping"
        )
    if mapping is not None and not isinstance(mapping, dict):
        raise Exception("Mapping must be a dict")

    if lbl_classes is None:
        lbl_classes = itertools.repeat(
            range(len(probs[0])), len(probs)
        )

    results = [
        {
            (
                mapping[str(lbl_class)]
                if mapping is not None
                else str(lbl_class)
            ): prob
            for lbl_class, prob in zip(*row)
        }
        for row in zip(lbl_classes, probs)
    ]

    return results


def get_layerwise_nodes(t, height_of_tree):
    root_node = t.name
    layerwise_nodes = {}
    for i in range(height_of_tree):
        node_list = []
        for node in t.traverse("preorder"):
            distance = t.get_distance(node.name, topology_only=True)

            if i == int(distance):
                node_list.extend([node.name])
        layerwise_nodes[i] = node_list

    return layerwise_nodes


def get_layerwise_predictions(layerwise_nodes, logits):

    number_of_layers = len(layerwise_nodes)
    layerwise_pred = []
    cnt = 0
    for i in range(number_of_layers):
        num_nodes = len(layerwise_nodes[i])
        layerwise_logit = logits[:, cnt : cnt + num_nodes]
        layerwise_pred.append(
            torch.nn.Softmax(dim=-1)(layerwise_logit)
        )
        cnt += num_nodes

    return layerwise_pred


def get_probability_tree(
    t, layerwise_pred, layerwise_nodes, batch_size, n_nodes, device
):

    node_prediction = {}
    for i in range(len(layerwise_nodes)):
        node_list = layerwise_nodes[i]
        prediction_list = layerwise_pred[i]
        for j in range(len(node_list)):
            node_prediction[node_list[j]] = prediction_list[:, j]

    node_probabilities = torch.zeros((batch_size, n_nodes))

    for node in t.traverse("preorder"):
        node_name = node.name
        prob = torch.ones(prediction_list[:, j].shape).to(device)
        while node.up:
            prob *= node_prediction[node.name]
            node = node.up

        node_probabilities[:, int(node_name)] = prob

    return node_probabilities


def toImgPIL(imgOpenCV):
    return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))


class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    topk = 1

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        # self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.profiler_args = {}

    """
    Base class for all vision handlers
    """

    def initialize(
        self,
        model_dir=None,
        serialized_file=None,
        limit_max_image_pixels=False,
    ):
        # super().initialize(context)

        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        # properties = context.system_properties
        if not limit_max_image_pixels:
            Image.MAX_IMAGE_PIXELS = None

        # self.manifest = context.manifest

        #  model_dir = properties.get("model_dir")

        # Load class mapping for classifiers
        mapping_file_path = os.path.join(
            model_dir, "index_to_name.json"
        )
        self.mapping = load_label_mapping(mapping_file_path)
        if "method" in list(self.mapping.keys()):
            self.method = self.mapping.pop("method")
        else:
            logging.error(
                "you must add a 'method' key (value in ['clustering', 'hierarchy',  'level'] in 'index_to_name.json' file"
            )

        if self.method == "level":
            if "height_of_tree" in list(self.mapping.keys()):
                self.height_of_tree = int(
                    self.mapping.pop("height_of_tree")
                )
            else:
                logging.error(
                    "you must add a 'height_of_tree' key (value is an integer) key in 'index_to_name.json' file with level method"
                )
            if "new_n_classes" in list(self.mapping.keys()):
                self.new_n_classes = int(
                    self.mapping.pop("new_n_classes")
                )
            else:
                logging.error(
                    "you must add a 'new_n_classes' key (value is int) key in 'index_to_name.json' file with level method"
                )
            if "ete_hierarchy" in list(self.mapping.keys()):
                self.ete_hierarchy = Tree(
                    self.mapping.pop("ete_hierarchy"), format=3
                )
                self.ete_hierarchy.name = str(self.new_n_classes - 1)
            else:
                logging.error(
                    "you must add a 'ete_hierarchy' key (value is string Tree) key in 'index_to_name.json' file with level method"
                )

        self.n_classes = len(self.mapping)
        model_pth_path = None
        # if "serializedFile" in self.manifest["model"]:
        #    serialized_file = self.manifest["model"]["serializedFile"]
        model_pth_path = os.path.join(model_dir, serialized_file)

        # model def file
        # model_file = self.manifest["model"].get("modelFile", "")

        # self.model = load_checkpoint(model_pth_path, self.device)

        self.model = torch.jit.load(model_pth_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Model file %s loaded successfully", model_pth_path)

        # if model_file:
        #     logger.debug("Loading eager model")
        #     self.model = load_checkpoint(model_pth_path, self.device)
        #     # self.model = self._load_pickled_model(
        #     #     model_dir, model_file, model_pth_path)
        #     self.model.to(self.device)
        # else:
        #     logger.debug("Loading torchscript model")
        #     if not os.path.isfile(model_pth_path):
        #         raise RuntimeError("Missing the model_pth file")

        #     self.model = self._load_torchscript_model(model_pth_path)

        # self.model.eval()
        if ipex_enabled:
            self.model = self.model.to(
                memory_format=torch.channels_last
            )
            self.model = ipex.optimize(self.model)

        logger.debug(
            "Model file %s loaded successfully", model_pth_path
        )

    def img_transforms(self):
        trans = []
        trans.append(transforms.Resize(size=(256, 256)))
        img_mean, img_std = (0.485, 0.456, 0.406), (
            0.229,
            0.224,
            0.225,
        )
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=img_mean, std=img_std))
        trans = transforms.Compose(trans)

        return trans

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """

        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row  # row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)
                print("isinstance(image, str)", type(image))

                image_transform = self.img_transforms()
                image = image_transform(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):

                image = Image.open(io.BytesIO(image))

                image_square = convert_to_square(np.array(image))

                image_transform = self.img_transforms()

                image = image_transform(toImgPIL(image_square))

            else:
                # if the image is a list
                image = torch.FloatTensor(image)
                image_transform = self.img_transforms()
                image = image_transform(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        marshalled_data = data.to(self.device)
        with torch.no_grad():
            results = self.model(marshalled_data, *args, **kwargs)

        return results

    def get_insights(self, tensor_data, _, target=0):
        return self.ig.attribute(
            tensor_data, target=target, n_steps=15
        ).tolist()

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):

        if self.method == "level":

            layerwise_nodes = get_layerwise_nodes(
                self.ete_hierarchy, self.height_of_tree
            )
            layerwise_pred = get_layerwise_predictions(
                layerwise_nodes, data
            )
            node_probabilities = get_probability_tree(
                self.ete_hierarchy,
                layerwise_pred,
                layerwise_nodes,
                data.shape[0],
                self.new_n_classes,
                self.device,
            )

            data = node_probabilities[:, : self.n_classes]

        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()

        return map_class_to_label(probs, self.mapping, classes)

    def handle(self, data):
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

        #  self.context = context

        # metrics = self.context.metrics

        data_preprocess = self.preprocess(data)

        if not self._is_explain():
            # inference_output = self.model(data_preprocess)
            x_emb, inference_output = self.inference(data_preprocess)

            output = self.postprocess(inference_output)

        stop_time = time.time()
        #  metrics.add_time(
        #     "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        # )

        return output

    def _is_explain(self):
        return False

