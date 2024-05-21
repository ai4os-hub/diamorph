#!/bin/bash
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

set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /home/model-server/config.properties --model-store /home/model-server/model-store --models all

elif [[ "$1" = "classif" ]]; then
    shift 1

    rm -rf model-store/*
    rm -rf Mobilenet.mar

    torch-model-archiver --model-name Mobilenet --version 1.0 --serialized-file best_model.tjm --handler handler.py --extra-files index_to_name.json

    (cd /home/model-server; printf "inference_address=http://0.0.0.0:${PORT_CLASS_INF}\nmanagement_address=http://0.0.0.0:${PORT_CLASS_MAN}\nmetrics_address=http://0.0.0.0:${PORT_CLASS_MET}\n" > config.properties)
ls -l /home/model-server    
    echo "Mobilenet.mar is created"
    mv Mobilenet.mar model-store
    echo "Mobilenet.mar moved in the share directory : model-store "
    torchserve --start --ts-config /home/model-server/config.properties --model-store /home/model-server/model-store --models all

elif [[ "$1" = "objectDetection" ]]; then
    shift 1

    rm -rf model-store/*
    rm -rf yolov5.mar

    torch-model-archiver --model-name yolov5 --version 1.0 --serialized-file best_model.pt --handler handler.py --extra-files extra-files/

    (cd /home/model-server; printf "inference_address=http://0.0.0.0:${PORT_OD_INF}\nmanagement_address=http://0.0.0.0:${PORT_OD_MAN}\nmetrics_address=http://0.0.0.0:${PORT_OD_MET}\n" > config.properties)

    echo "yolov5.mar created in container"
    mv yolov5.mar model-store
    
    echo "yolov5.mar moved in the share directory : model-store "

    torch-model-archiver --model-name yolov5 --version 1.0 --serialized-file best_model.pt --handler handler.py --extra-files extra-files/
    

    torchserve --start --ts-config /home/model-server/config.properties --model-store /home/model-server/model-store --models all


elif [[ "$1" = "client" ]]; then

    streamlit run main.py
    
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
