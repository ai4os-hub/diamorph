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

import requests
import numpy as np
from PIL import Image

import cv2
import os
import matplotlib
import torch
import time

import shapely.geometry
import rasterio.features
from io import BytesIO

import streamlit as st
import shutil

Time = time.time()
# Settings
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only

PI = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


# -----------------------------------
# ------  USEFUL FUNCTION------------
# -----------------------------------


def to_img_pil(img_open_cv):
    return Image.fromarray(cv2.cvtColor(img_open_cv, cv2.COLOR_BGR2RGB))


def load_image(image_file):
    img = Image.open(image_file)
    return img


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def clean_streamlit_folder():
    st.legacy_caching.clear_cache()
    if os.path.exists("file.zip"):
        os.remove("file.zip")
    if os.path.isdir(os.getcwd() + "/__pycache__"):
        shutil.rmtree(os.getcwd() + "/__pycache__")
    for root, dirs, files in os.walk(os.getcwd() + "/tmp"):
        for file in files:
            os.unlink(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))


def convert_pil_inbyte_arr(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def get_boxes_and_taxa(image_file, is_pil=False, view_image=True):
    # To View Uploaded Image
    if is_pil:
        pil_input_image = image_file
    else:
        pil_input_image = load_image(image_file)
    pil_input_image = pil_input_image.convert("RGB")

    # Resize the image so that the largest side is 512x512
    # This must correspond to object_detection/extra-files/detect_params.yml:imgsz
    pil_input_image = expand2square(pil_input_image, 0)
    pil_input_image = pil_input_image.resize((512, 512))

    if view_image:
        st.image(pil_input_image, width=250)

    # get boxes and taxa from object detection inference port
    od_port = os.environ["PORT_OD_INF"]

    output_cv2image, crops_cv2, response, ecotaxas = client_request(
        np.array(pil_input_image),
        od_pred_url="http://objectDetection:" + od_port + "/predictions/yolov5",
    )

    # To View Image with boxes and taxa
    if view_image:
        st.image(to_img_pil(output_cv2image), width=250)

    # Original image came from cv2 format, fromarray convert into PIL format
    result = Image.fromarray(output_cv2image)
    crops = [Image.fromarray(crop) for crop in crops_cv2]

    return result, crops, response, ecotaxas


def rotate_plot_crops(xywh, real, imagin, im0s, it_obj, save_path=False):
    """
    Plots the croped images in bounding box using OpenCV
    """

    a_sin = torch.asin(imagin)
    a_cos = torch.acos(real)
    if imagin < 0:
        if abs(imagin) < PI / 4:
            angle = int(a_sin / PI * 180)
        else:
            angle = int(-a_cos / PI * 180)

    else:
        angle = int(a_sin / PI * 180)

    xywhrm = torch.FloatTensor([xywh + [real, imagin]])
    xyxyxyxy = xywhrm2xyxyxyxy(xywhrm)
    poly = shapely.geometry.Polygon(xyxyxyxy.view(4, 2))
    mask = rasterio.features.rasterize([poly], out_shape=im0s.shape[0:2]).astype("bool")
    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
    masked_img = mask * im0s
    # imutils.rotate(masked_im0, angle=int(angle))
    # sum              (vectorized c loop, fastest)
    sum_ = np.sum(masked_img, axis=2)
    non_black = sum_ != 0  # comparison       (vectorized c loop, fast)
    x_sc, y_sc = np.where(non_black)  # scan & convert   (c loop, slower)
    crop_img = masked_img[x_sc.min() + 1 : x_sc.max(), y_sc.min() + 1 : y_sc.max()]

    height, width = crop_img.shape[:2]
    #     quarter_height, quarter_width = height / 2, width / 2
    T = np.float32([[1, 0, 1.5 * width], [0, 1, 1.5 * height]])
    img_translation = cv2.warpAffine(crop_img, T, (width * 4, height * 4))

    centerx = width * 2
    centery = height * 2

    rotationMatrix = cv2.getRotationMatrix2D((centerx, centery), angle, 1.0)

    rotatingimage = cv2.warpAffine(
        img_translation, rotationMatrix, (width * 4, height * 4)
    )  # (xywh[3],xywh[2]) )  #  (masked_img.shape[0], masked_img.shape[1]))

    # sum              (vectorized c loop, fastest)
    sum_ = np.sum(rotatingimage, axis=2)
    non_black = sum_ != 0  # comparison       (vectorized c loop, fast)
    x_sc, y_sc = np.where(non_black)  # scan & convert   (c loop, slower)
    crop_img_new = rotatingimage[
        x_sc.min() + 1 : x_sc.max(), y_sc.min() + 1 : y_sc.max()
    ]

    crop_img = crop_img_new
    height, width = crop_img.shape[:2]
    if height < width:  # image is horizontal : must be turn vertical

        #     quarter_height, quarter_width = height / 2, width / 2
        T = np.float32([[1, 0, 1.5 * width], [0, 1, 1.5 * height]])
        img_translation = cv2.warpAffine(crop_img, T, (width * 4, height * 4))
        centerx = width * 2
        centery = height * 2
        rotationMatrix = cv2.getRotationMatrix2D((centerx, centery), 90, 1.0)
        rotatingimage = cv2.warpAffine(
            img_translation, rotationMatrix, (width * 4, height * 4)
        )  # (xywh[3],xywh[2]) )  #  (masked_img.shape[0], masked_img.shape[1]))
        # sum              (vectorized c loop, fastest)
        sum_ = np.sum(rotatingimage, axis=2)
        non_black = sum_ != 0  # comparison       (vectorized c loop, fast)
        x_sc, y_sc = np.where(non_black)  # scan & convert   (c loop, slower)
        crop_img_new = rotatingimage[
            x_sc.min() + 1 : x_sc.max(), y_sc.min() + 1 : y_sc.max()
        ]

    if save_path:
        save_crop_name = (
            os.path.dirname(save_path)
            + "/"
            + os.path.basename(save_path).split(".")[0]
            + "_th{}.png".format(it_obj)
        )
        cv2.imwrite(save_crop_name, crop_img_new)

    it_obj += 1

    height_im0s, width_im0 = im0s.shape[:2]
    [x_c, y_c, w_b, h_b] = [k.item() for k in xywh]
    object_integrity = True
    if x_c + w_b > width_im0:
        # print("\nx_c + w_b > width_im0")
        # print(
        #     {
        #         "img_file_name": os.path.basename(save_crop_name),
        #         "object_id": os.path.basename(save_crop_name).split(".")[0],
        #         "object_x_px": xywh[0].item(),
        #         "object_y_px": xywh[1].item(),
        #         "object_width_px": xywh[2].item(),
        #         "object_height_px": xywh[3].item(),
        #         "object_angle_deg": angle,
        #         "object_integrity": object_integrity
        #     }
        # )
        object_integrity = False
    if x_c - w_b < 0:
        # print("\nx_c - w_b < 0")
        # print(
        #     {
        #         "img_file_name": os.path.basename(save_crop_name),
        #         "object_id": os.path.basename(save_crop_name).split(".")[0],
        #         "object_x_px": xywh[0].item(),
        #         "object_y_px": xywh[1].item(),
        #         "object_width_px": xywh[2].item(),
        #         "object_height_px": xywh[3].item(),
        #         "object_angle_deg": angle,
        #         "object_integrity": object_integrity
        #     }
        # )

        object_integrity = False
    if y_c + h_b > height_im0s:
        # print("\ny_c + h_b > height_im0s")
        # print(
        #     {
        #         "img_file_name": os.path.basename(save_crop_name),
        #         "object_id": os.path.basename(save_crop_name).split(".")[0],
        #         "object_x_px": xywh[0].item(),
        #         "object_y_px": xywh[1].item(),
        #         "object_width_px": xywh[2].item(),
        #         "object_height_px": xywh[3].item(),
        #         "object_angle_deg": angle,
        #         "object_integrity": object_integrity
        #     }
        # )
        object_integrity = False
    if y_c - h_b < 0:
        # print("\ny_c -h_b < 0 ")
        # print(
        #     {
        #         "img_file_name": os.path.basename(save_crop_name),
        #         "object_id": os.path.basename(save_crop_name).split(".")[0],
        #         "object_x_px": xywh[0].item(),
        #         "object_y_px": xywh[1].item(),
        #         "object_width_px": xywh[2].item(),
        #         "object_height_px": xywh[3].item(),
        #         "object_angle_deg": angle,
        #         "object_integrity": object_integrity
        #     }
        # )
        object_integrity = False

    if save_path:

        return {
            "img_file_name": os.path.basename(save_crop_name),
            "object_id": os.path.basename(save_crop_name).split(".")[0],
            "object_x_px": xywh[0].item(),
            "object_y_px": xywh[1].item(),
            "object_width_px": xywh[2].item(),
            "object_height_px": xywh[3].item(),
            "object_angle_deg": angle,
            "object_integrity": object_integrity,
        }
    else:
        return {
            "crop_img_new": crop_img_new,
            "object_id": "{}".format(it_obj),
            "object_x_px": xywh[0].item(),
            "object_y_px": xywh[1].item(),
            "object_width_px": xywh[2].item(),
            "object_height_px": xywh[3].item(),
            "object_angle_deg": angle,
            "object_integrity": object_integrity,
        }


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


# Ancillary functions with rotate anchor boxes-------------
def rotate_plot_one_box(x_pix, im, color=(128, 128, 128), label=None, line_thickness=3):
    """
    Plots one bounding box on image 'im' using OpenCV
    im is np.array with shape (W, H, Ch), x is pixel-level xywhrm
    """
    xyxyxyxy = xywhrm2xyxyxyxy(x_pix)
    polygon_plot_one_box(xyxyxyxy, im, color, label, line_thickness)


# Ancillary functions with polygon anchor boxes-------------
def polygon_plot_one_box(
    x_pix, im_arr, color=(128, 128, 128), label=None, line_thickness=3
):
    """
    Plots one bounding box on image 'im' using OpenCV
    im is np.array with shape (W, H, Ch), x_pix is pixel-level xyxyxyxy
    """

    assert (
        im_arr.data.contiguous
    ), "Image not contiguous. Apply np.ascontiguousarray(im) to polygon_plot_one_box() input image."
    tl = (
        line_thickness or round(0.002 * (im_arr.shape[0] + im_arr.shape[1]) / 2) + 1
    )  # line/font thickness
    c = (
        x_pix.cpu().numpy().reshape(-1, 1, 2).astype(np.int32)
        if isinstance(x_pix, torch.Tensor)
        else np.array(x_pix).reshape(-1, 1, 2).astype(np.int32)
    )
    cv2.polylines(
        im_arr, pts=[c], isClosed=True, color=color, thickness=tl, lineType=cv2.LINE_AA
    )
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c_1 = (int(c[:, 0, 0].mean()), int(c[:, 0, 1].mean()))
        c_2 = (int(c_1[0] + t_size[0]), int(c_1[1] - t_size[1] - 3))
        im_origin = im_arr.copy()
        cv2.rectangle(im_arr, c_1, c_2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im_arr,
            label,
            (c_1[0], c_1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
        alpha = 0.5  # opacity of 0.5
        cv2.addWeighted(im_arr, alpha, im_origin, 1 - alpha, 0, im_arr)


# Ancillary functions with rotate anchor boxes-------------
def xywhrm2xyxyxyxy(xywhrm):
    """
    xywhrm : shape (N, 6)
    Transform x,y,w,h,re,im to x_1,y_1,x2,y2,x3,y3,x4,y4
    Suitable for both pixel-level and normalized
    """
    is_array = isinstance(xywhrm, np.ndarray)
    if is_array:
        xywhrm = torch.from_numpy(xywhrm)

    x_0, x_1, y_0, y_1 = (
        -xywhrm[:, 2:3] / 2,
        xywhrm[:, 2:3] / 2,
        -xywhrm[:, 3:4] / 2,
        xywhrm[:, 3:4] / 2,
    )
    xyxyxyxy = (
        torch.cat((x_0, y_0, x_1, y_0, x_1, y_1, x_0, y_1), dim=-1)
        .view(-1, 4, 2)
        .contiguous()
    )
    R = torch.zeros(
        (xyxyxyxy.shape[0], 2, 2), dtype=xyxyxyxy.dtype, device=xyxyxyxy.device
    )
    R[:, 0, 0], R[:, 1, 1] = xywhrm[:, 4], xywhrm[:, 4]
    R[:, 0, 1], R[:, 1, 0] = xywhrm[:, 5], -xywhrm[:, 5]

    xyxyxyxy = (
        torch.matmul(xyxyxyxy, R).view(-1, 8).contiguous()
        + xywhrm[:, [0, 1, 0, 1, 0, 1, 0, 1]]
    )
    return xyxyxyxy.cpu().numpy() if is_array else xyxyxyxy


def client_request(
    input, od_pred_url="http://localhost:9080/predictions/yolov5"  # cv2 array
):

    label_cls = [
        "AAMB",
        "ACLI",
        "ADAM",
        "ADCT",
        "ADEU",
        "ADLA",
        "ADMI",
        "ADMO",
        "ADPY",
        "ADRI",
        "ADRU",
        "ADSB",
        "ADSH",
        "ADSU",
        "AFOR",
        "AMID",
        "AOVA",
        "APED",
        "AUGA",
        "AUGR",
        "AUPU",
        "AUSU",
        "CAEX",
        "CAGR",
        "CATO",
        "CDUB",
        "CEUG",
        "CINV",
        "CLCT",
        "CLNT",
        "CMED",
        "CMEN",
        "CMLF",
        "COPL",
        "CPED",
        "CPLA",
        "CRAC",
        "CSLP",
        "CTPU",
        "DCOF",
        "DMES",
        "DMON",
        "DOCU",
        "DSTE",
        "DTEN",
        "DVUL",
        "ECAE",
        "ECPM",
        "EICD",
        "EMIN",
        "ENCM",
        "ENMI",
        "ENVE",
        "EOCO",
        "ESLE",
        "ESUM",
        "ETEN",
        "FCRO",
        "FFVI",
        "FGRA",
        "FLEN",
        "FMES",
        "FMOC",
        "FNEV",
        "FPEC",
        "FPRU",
        "FSAP",
        "FSBH",
        "FSLU",
        "FVAU",
        "GACC",
        "GBOB",
        "GCLF",
        "GCUN",
        "GELG",
        "GLAT",
        "GMIN",
        "GOLI",
        "GPAR",
        "GPLI",
        "GPRI",
        "GRHB",
        "GSCI",
        "GTER",
        "GTRU",
        "HARC",
        "HLMO",
        "HVEN",
        "KALA",
        "KCLE",
        "LGOE",
        "MCCO",
        "MCIR",
        "MVAR",
        "NACD",
        "NACI",
        "NAMP",
        "NANT",
        "NCPL",
        "NCPR",
        "NCRY",
        "NCTE",
        "NCTO",
        "NCTV",
        "NDIS",
        "NERI",
        "NFIL",
        "NFON",
        "NGER",
        "NGRE",
        "NHEU",
        "NIBU",
        "NIPF",
        "NIPU",
        "NLAN",
        "NLIN",
        "NMIC",
        "NPAE",
        "NPAL",
        "NRAD",
        "NRCH",
        "NRCS",
        "NROS",
        "NSOC",
        "NSTS",
        "NSUA",
        "NTPT",
        "NTRV",
        "NULA",
        "NVEN",
        "NYCO",
        "NZSU",
        "PAPR",
        "PBIO",
        "PDAO",
        "PGRN",
        "PHEL",
        "PLAU",
        "PLFR",
        "PROH",
        "PSAT",
        "PSBR",
        "PSXO",
        "PTCO",
        "PTDE",
        "PTDU",
        "PTLA",
        "PULA",
        "RABB",
        "RSIN",
        "RUNI",
        "SBKU",
        "SBND",
        "SCON",
        "SHTE",
        "SIDE",
        "SKPO",
        "SLAC",
        "SPUP",
        "SSVE",
        "TAPI",
        "TFAS",
        "TGES",
        "THLA",
        "TLEV",
        "UULN",
        " ",
    ]
    line_thickness = 3

    url = od_pred_url
    image = input

    image_base = image.copy()

    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    byte_im = im_buf_arr.tobytes()

    time_response = time.time()

    try:
        response = requests.post(url, data=byte_im).json()  # github api
        print("response status : 200")
    except requests.exceptions.HTTPError as e:
        assert e, "Error: " + str(e)

    print("Time for inference: " + str((time.time() - time_response)))

    colors = Colors()  # create instance for 'from utils.plots import colors'

    it_obj = 0
    crops = []
    ecotaxas = []

    if not (isinstance(response, dict) and response["message"] == "Prediction failed"):
        for det in response:
            xywh = [det["x"], det["y"], det["w"], det["h"]]
            real, imagin, conf, cls = [
                det["real"],
                det["imagin"],
                det["class confidence"],
                det["class"],
            ]

            im = rotate_plot_one_box(
                torch.tensor([*xywh, real, imagin]).cpu().view(-1, 6).numpy(),
                image,
                label=cls + " {:.2f}".format(conf),
                color=colors(label_cls.index(cls), True),
                line_thickness=line_thickness,
            )

            # ------------
            xywh_tensor = [torch.tensor(k) for k in xywh]
            crop = rotate_plot_crops(
                xywh_tensor,
                torch.tensor(real),
                torch.tensor(imagin),
                image_base,
                it_obj,
            )

            crops.append(crop["crop_img_new"])

            ecotaxa = rotate_plot_crops(
                xywh_tensor,
                torch.tensor(real),
                torch.tensor(imagin),
                image_base,
                it_obj,
                save_path=str(it_obj),
            )

            ecotaxas.append(ecotaxa)

    return image, crops, response, ecotaxas
