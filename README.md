# Maat

![](https://img.shields.io/badge/version-1.0-blue.svg) 
![](https://img.shields.io/badge/language-python-orange.svg)

![maat-7599155_1280](https://github.com/BEbillionaireUSD/Maat/assets/49298462/afbfbc27-7853-40a0-ad00-90efac5d58aa)

Here is the repo for our paper ``Maat: Performance Metric Anomaly Anticipation for Cloud Services with Conditional Diffusion'', accepted by ASE 2023.

## Architecture

#### Overview

![Maat 001](https://user-images.githubusercontent.com/112700133/216756546-635088cd-c61b-4fa7-88e6-6979eb8df4c3.jpeg)

#### Diffusion Model

![Maat 002](https://user-images.githubusercontent.com/112700133/216756574-f1f6d2a4-93ab-4fa2-a5be-106e5f9922d8.jpeg)

#### Model M

![Maat 003](https://user-images.githubusercontent.com/112700133/216756602-736d7dc2-9253-44e6-83ab-f13f8f188733.jpeg)

## Data
The data should be stored in csv files with the first column being ``timestamp'' and the last column being ``label''. If labels are not avaliable, it can be all zeros.

We put an example dataset (part of the AIOps18 dataset due to the space limit) in the ``data'' director.

## Tree
```
.
├── dataload.py
├── detect.py
├── extract_feat.py
├── model
│   ├── PixelCNN.py
│   ├── detection.py
│   ├── diffusion.py
│   ├── network.py
│   ├── prediction.py
│   └── util.py
├── predict.py
├── requirements.txt
└── util.py
```

## Environment
We support python3.x $\geq$ 3.7. The environment can be built by:
```$ pip install -r requirements.txt```

## Running
For the prediction phase, run:
```$ cd src && python predict.py```

For the detection phase, you should first extract features and then run:
```$ cd src && python detect.py```


