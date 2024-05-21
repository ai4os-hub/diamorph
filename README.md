# diamorph-prediction
An application to detect and predict diatoms on microscopic images. The code is taken from "diamorph_app".


The model used for this application is trained with the code at [https://github.com/vaishwarya96/diatom_codes](https://github.com/vaishwarya96/diatom_codes)


## Content

```
├── docker-compose.yml
├── Makefile
├── .env.sample
├── .gitignore
├── README.md

├── classification
│   ├── best_model
│   │   ├── best_model.tjm
│   │   └── index_to_name.json
│   ├── BRG_AAMB_10002.png
│   ├── config.properties
│   ├── handler.py

├── client
│   ├── client_request.py
│   ├── main.py
│   └── st_button.py

├── diamorph_image
│   ├── dockerd-entrypoint.sh
│   ├── Dockerfile
│   └── requirements.txt

├── object_detection
│   ├── 43.png
│   ├── 68.png
│   ├── 80.png
│   ├── best_model
│   │   └── best_model.pt
│   ├── config.properties
│   ├── extra-files
│   │   ├── detect_params.yaml
│   │   ├── models/*
│   │   └── utils/*
│   ├── handler.py
│   ├── images.zip
```

##  Installation

- If it is not done yet, intall `docker`,  `docker compose`, and add your user name in `docker` with the command 
```
sudo adduser "your_user_name" docker
```
    
and reboot. The current code has been tested on ubuntu 22.04, docker compose v2.27.0, docker community edition 26.1.2

### Classification
```
├── classification
│   ├── best_model
│   │   ├── best_model.tjm
│   │   └── index_to_name.json
```

We suppose you have succesfully trained a classification model. You need to
import, from your trained classifier the following files :

- put the *array classes* and the *method* (e.g. "method": "clustering") defined in `classification/best_model/index_to_name.json`
- put the weights of the classification model into `classification/best_model/best_model.tjm`

### Object detection

```
├── object_detection
│   ├── best_model
│   │   ├── best_model.pt
│   ├── extra-files
│   │   ├── detect_params.yaml
```

We suppose you have trained an object detection model from which you need to
import the following files :

- put the inference parameter for object detection in `object_detection/extra-files/detect_params.yaml`
- put the weights of the object detection model `object_detection/best_model/best_model.pt` 

### Docker image build and run

- Build the recipe (contruct `diamorph:latest` docker image + create `.mar` files cf *# Inside deployment* ) typing :

```
make build
```

- Run the recipe typing : 

```
make up
```

## Test the deployment

- Go to the url hosting the docker from a browser : http://localhost:5002  

The `5002` port is the default port we specified in the `.env.sample` file. If you changed it, you need to adapt the URL accordingly

- click on button `Image` (on the left) and upload `object_detection/68.png`. You should get :

```
{
"filename":"68.png"
"filetype":"image/png"
"filesize":153258
}
```
- The plot of the image. 

- Few seconds later, the same plot with boxes and taxa below

- And a button : `Donwload ZIP boxes & taxa`

- Click on the button `Donwload ZIP boxes & taxa`. you should  be able to donwload `file.zip` with in : 

	```
	├── 68_bt0_FCRO.png
	├── 68_bt1_FCRO.png
	├── 68_bt2_FCRO.png
	├── 68_bt3_FLEN.png
	├── 68_bt4_NCTE.png
	├── 68_bt5_RUNI.png
	├── 68_bt6_PLFR.png
	├── 68_bt7_ADSB.png
	├── 68_bt.png
	├── boxes_taxa.json
	└── ecotaxa.json
	```

- click on button `Zip` (on the left) and upload `object_detection/images.zip` (with `68.png` and `43.png` inside). You should get 

```
{
"filename":"images.zip"
"filetype":"application/zip"
"filesize":318809
}
```
- A progression blue bar, and when it finish

- And a button : `Donwload ZIP boxes & taxa` 

- Click on the button `Donwload ZIP boxes & taxa`. you should be able to donwload file.zip with in : 

```
├── 43_bt0_FCRO.png
├── 43_bt1_ECAE.png
├── 43_bt2_RUNI.png
├── 43_bt3_FCRO.png
├── 43_bt4_FCRO.png
├── 43_bt5_FCRO.png
├── 43_bt6_FCRO.png
├── 43_bt7_ENVE.png
├── 43_bt8_FSAP.png
├── 43_bt9_SPUP.png
├── 43_bt.json
├── 43_bt.png
├── 68_bt0_FCRO.png
├── 68_bt1_FCRO.png
├── 68_bt2_FCRO.png
├── 68_bt3_FLEN.png
├── 68_bt4_NCTE.png
├── 68_bt5_RUNI.png
├── 68_bt6_PLFR.png
├── 68_bt7_ADSB.png
├── 68_bt.json
├── 68_bt.png
└── ecotaxa.json
```

## Customization of the ports

The diamorph app relies on three running docker containers : 
- one for object detection, 
- one for classification,
- one for the frontend client.

All these containers communicate through the network with dedicated ports. These ports are specified in the `.env.sample` file. 

If you need to customize these ports, modify the `.env.sample` file, remove the generated `.env` file and call `make`. This will recreate the `.env` file. 

The port for connecting to the web frontend is specified by the variable `LOCAL_PORT`. It is set by default to `5002`. This is the port on the host running for the docker images that will be used by the application. All the other ports are internal to the docker container.

## Inside deployment

The *`.env.sample`* file define the environnment variables by default for running the app. *`.env`* and *`.envrc`* are created by the makefile. It should adapt to every machine configurations. 

- `make build` : build the docker image : 
    - It creates in *`diamorph_image`* folder the diamorph docker image with   
        - an entrypoint *`diamorph_image/dockerd-entrypoint.sh`* with commands [`classif` , `objectDetection`, `client`] to run for each docker-compose services. 
        - Also, the entrypoint creates *`config.properties`* files for *torchserve* to deploy as a service `classif` and `objectDetection` models
        - For `classif` : 
            - The entrypoint *`diamorph_image/dockerd-entrypoint.sh`* creates a file  *`Mobilenet.mar`* by running torchserve on [*`classification/handler.py`*, *`classification/best_model/best_model.tjm`*, *`classification/best_model/index_to_name.json`*].
            - *`Mobilenet.mar`* is stored *`classification/model-store`*. The *`Mobilenet.mar`* file is used to deploy the model. If you want to update inference by changing *`classification/best_model/best_model.pt`* and *`classification/best_model/index_to_name.json`*, you have to rerun the recipe (make build) to take into account the new  models and weights. 
        - For `object_detection` : 
            - The recipe creates the file *`yolov5.mar`* by running torchserve on [*`object_detection/handler.py`*, *`object_detection/best_model/best_model.pt`*, *`object_detection/extra-files/detect_params.yaml`*, *`extra-files/models/*`*, *`extra-files/utils/*`*]
            - *`yolov5.mar`* is stored in *`object_detection/model-store`*. The *`yolov5.mar`* file is used to deploy the model. If you want to update inference by changing *`object_detection/best_model/best_model.pt`* and *`object_detection/extra-files/detect_params.yaml`*, you have to rerun the recipe (make build) to take into account the new weights. 

-   `make up` : run docker compose up serve models `objectDetection`, `classif` and run the service `client`
    - `client` works with `streamlit` : 
        - upload a (button=`Image`) or several (button=`Zip`)  microscopic diatom image(s) from the user
        - send image(s) to objectDetection service :
            - get boxes, extract thumnblnails 
                - send  thumnblnails to `classif` service to get the taxa. 
            - return to `client` json with boxes and taxa
        - download results (and plot them if button=`Image` )

## Authors and maintaining

Cyril Regan, Aishwarya Venkataramanan, Jeremy Fix, Martin Laviale, Cédric Pradalier

If you need some information about this code, please contact Jeremy Fix (jeremy.fix@centralesupelec.fr) and Martin Laviale (martin.laviale@univ-lorraine.fr)

## License

This software is released under the AGPL license; See [Here](https://www.gnu.org/licenses/agpl-3.0.txt)

