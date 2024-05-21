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


import streamlit as st

from st_button import button_image, button_zip

# -----------------------------------------------------------------------------------------------
# ----------------------------------------     APP     ------------------------------------------
# -----------------------------------------------------------------------------------------------

# set title of the App
st.title("Diatom object detection")

# set menu of the App
menu = ["Image", "Zip"]
choice = st.sidebar.selectbox("Menu", menu)


if choice == "Image":  # upload only one image
    button_image()

elif choice == "Zip":  # upload a zip dir of images
    button_zip()

st.markdown(
    """
This application has been developed thanks to the support of [PNRIA program](https://www.intelligence-artificielle.gouv.fr/fr/thematiques/programme-national-de-recherche-intelligence-artificielle-pnria) with INS2I.\n\n
The application has been developed by Cyril Regan (SISR - Loria/Inria Grand-Est) based on an early work of Aishwarya Venkataramanan during her PhD thesis at Georgia Tech CNRS.

The PhD scholarship of Aishwarya is funded by ANR [ANR-20-THIA-0010] and Région Grand-Est. Additional financial support was
provided by CNRS (ZAM LTSER Moselle, PNRIA) and Université de Lorraine (LUE).

It is released under the AGPLv3 license. See [https://www.gnu.org/licenses/agpl-3.0.txt](https://www.gnu.org/licenses/agpl-3.0.txt).
"""
)
