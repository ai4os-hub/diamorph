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

import argparse

import yaml

from wandb_utils import WandbLogger

WANDB_ARTIFACT_PREFIX = "wandb-artifact://"


def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data = yaml.safe_load(f)  # data dict
    logger = WandbLogger(opt, "", None, data, job_type="Dataset Creation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="train as single-class dataset"
    )
    parser.add_argument(
        "--project", type=str, default="YOLOv5", help="name of W&B Project"
    )
    opt = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job

    create_dataset_artifact(opt)
