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

"""Perform test request"""
import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_IMAGE = "zidane.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
