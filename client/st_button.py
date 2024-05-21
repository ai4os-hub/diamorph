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

from copyreg import pickle
from PIL import Image

import os
import time

import streamlit as st

import zipfile
import json


# images extension
INPUT_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")
OUPUT_EXTENSIONS = "png"

from client_request import (
    convert_pil_inbyte_arr,
    get_boxes_and_taxa,
    clean_streamlit_folder,
)


def results_in_zip_image(
    response,
    zip_name,
    jsonString,
    image_file_output,
    img_byte_arr,
    img_byte_arr_crop,
    ecotaxas,
):
    """_summary_
        write results from the image prediction in zip file
    Args:
        response (list of dict): object detected with taxa
        zip_name (string): name of the zip file
        jsonString (string): json convertion of `response`
        image_file_output (string): image file name
        img_byte_arr (byte): PIL array conversion of the image (with boxes and taxas)
        img_byte_arr_crop (byte): PIL array conversion of the cropped images
            (extraction of boxes in initial images)
        ecotaxas (string): json string for `https://ecotaxa.obs-vlfr.fr/` export
    """
    # store the json and the image in zip file
    with zipfile.ZipFile(zip_name, "w") as zip:
        zip.writestr("boxes_taxa.json", jsonString)
        zip.writestr(image_file_output, img_byte_arr)

        for ii, image_file_crop in enumerate(img_byte_arr_crop):
            crop_name = (
                ".".join(image_file_output.split(".")[:-1])
                + str(ii)
                + "_"
                + response[ii]["class"]
                + "."
                + OUPUT_EXTENSIONS
            )
            zip.writestr(crop_name, image_file_crop)

        zip.writestr(crop_name, image_file_crop)
        zip.writestr("ecotaxa.json", json.dumps(ecotaxas))

    # Parse made zip file to streamlit button
    with open(zip_name, "rb") as fp:
        btn = st.download_button(
            label="Download ZIP boxes & taxa",
            data=fp,
            file_name=zip_name,
            mime="application/octet-stream",
        )


def results_in_zip_zip(results):
    """_summary_
        write [results] from the images predictions in zip file

    Args:
        results (list of list): [[image_file_output, result, crops, response, ecotaxas]]
    """

    timestr = time.strftime("%Y%m%d_%H%M%S")
    zip_name = f"file{timestr}.zip"
    with zipfile.ZipFile(zip_name, "w") as zip:
        ecotaxas_concat = []
        for i, (image_name, pil_image, crops, response, ecotaxas) in enumerate(results):

            ecotaxas_concat = ecotaxas_concat + ecotaxas
            img_byte_arr = convert_pil_inbyte_arr(pil_image)
            zip.writestr(image_name + "." + OUPUT_EXTENSIONS, img_byte_arr)
            zip.writestr(image_name + ".json", json.dumps(response))

            img_byte_arr_crop = [convert_pil_inbyte_arr(crop) for crop in crops]
            for ii, image_file_crop in enumerate(img_byte_arr_crop):
                crop_name = (
                    image_name
                    + str(ii)
                    + "_"
                    + response[ii]["class"]
                    + "."
                    + OUPUT_EXTENSIONS
                )
                zip.writestr(crop_name, image_file_crop)

        zip.writestr("ecotaxa.json", json.dumps(ecotaxas_concat))
    # Parse made zip file to streamlit button
    with open(zip_name, "rb") as fp:
        btn = st.download_button(
            label="Download ZIP boxes & taxa",
            data=fp,
            file_name=zip_name,
            mime="application/octet-stream",
        )


def button_image():
    """_summary_
    - Create a button to upload an image (png) and print it
    - Get_boxes_and_taxa from `object_detection` (and `classif`) services, and print it
    - Create download button to store on client device the results
    """

    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=INPUT_EXTENSIONS)

    if image_file is not None:

        # To See details
        file_details = {
            "filename": image_file.name,
            "filetype": image_file.type,
            "filesize": image_file.size,
        }
        st.write(file_details)

        # get boxes and taxa from object detection and classification port
        result, crops, response, ecotaxas = get_boxes_and_taxa(image_file)
        print("response st button", result, crops, response, ecotaxas)
        if crops == []:
            st.write(
                "**:x: : Sorry, no object found for `{}`**.  Did you load the good image ?".format(
                    file_details["filename"]
                )
            )
        else:

            # define output filename
            image_file_output = (
                ".".join(image_file.name.split(".")[:-1])
                + "_bt"
                + "."
                + OUPUT_EXTENSIONS
            )

            # convert result PIL image in a byte array buffer
            img_byte_arr = convert_pil_inbyte_arr(result)
            img_byte_arr_crop = [convert_pil_inbyte_arr(crop) for crop in crops]

            # convert the response dict into a json string
            jsonString = json.dumps(response)

            timestr = time.strftime("%Y%m%d_%H%M%S")
            zip_name = f"file{timestr}.zip"

            results_in_zip_image(
                response,
                zip_name,
                jsonString,
                image_file_output,
                img_byte_arr,
                img_byte_arr_crop,
                ecotaxas,
            )

        # clean tmp files in server
        clean_streamlit_folder()


def button_zip():
    """_summary_
    - Create a button to upload an zip of images (png)
    - Get_boxes_and_taxa from `object_detection` (and `classif`) services, for each images
    - Create download button to store on client device the results
    """

    st.subheader("Zip")

    image_zip = st.file_uploader("Upload Images", type="zip")

    if image_zip is not None:
        # -------------    READ zip file
        # To See details
        file_details = {
            "filename": image_zip.name,
            "filetype": image_zip.type,
            "filesize": image_zip.size,
        }
        st.write(file_details)
        # extract zip content in "tmp" dir
        try:
            with zipfile.ZipFile(image_zip) as z:
                z.extractall("tmp")
        except:
            st.write("Invalid file")

        # -------------    READ zip file and get boxes and taxa
        results = []
        it = 1

        my_bar = st.progress(0)

        for root, dirs, files in os.walk(
            os.getcwd() + "/tmp"
        ):  # get files from 'tmp' folder
            for f in files:
                # read with PIL the image
                pil_image = Image.open(os.path.join(root, f))

                # get boxes and taxa from object detection and classification port
                result, crops, response, ecotaxas = get_boxes_and_taxa(
                    pil_image, is_pil=True, view_image=False
                )

                if response == []:
                    st.write("**:x: : no object found for `{}`**".format(f))
                else:
                    # define output filename
                    image_file_output = ".".join(f.split(".")[:-1]) + "_bt"

                    # store output filename, image and json response in 'results' list
                    results.append(
                        [image_file_output, result, crops, response, ecotaxas]
                    )

                    my_bar.progress(it / len(z.namelist()))
                    it = +1

            my_bar.progress(1.00)

        # -------------    WRITE zip file

        results_in_zip_zip(results)

        # clean tmp files in server
        clean_streamlit_folder()
