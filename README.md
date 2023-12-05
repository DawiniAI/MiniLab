
# MiniLab | From Dawini.ai Team

The first tool to automate the process of analyzing lab tasks data through Deep Learning

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)


## Acknowledgements
 - [ZiadOmar@Kaggle](https://www.kaggle.com/code/zeadomar/breast-cancer-detection-with-cnn/comments)
 - [A.MOHAN.KUMAR@Kaggle](https://www.kaggle.com/code/amohankumar/bone-break-classifier-using-cnn)


## Features

- Prediction Percentage of Cancer 
- Prediction Class of Fractions


## Installation
1- install the requirments:
```bash
pip install -r requirments.txt
```

2- Run Flask Server :
```python 
python app.py

```
output:

```bash
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```


## Authors

- [@loaiabdalslam](https://www.github.com/loaiabdalslam)

- [@solimanware](https://github.com/solimanware)

## API Reference

#### Get breast perdiction

```http
  POST /berast/predict
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `img` | `file` | **Required**.  |

- Response
```http
{
    "cancer_percentage": 95.46,
    "not_cancer_percentage": 4.54
}
```



#### Get bone perdiction

```http
  POST /bone/predict
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `img` | `file` | **Required**.  |

- Response
```http
  {
    "avulsion_fracture": 98.09,
    "comminuted_fracture": 0.12,
    "compression_crush_fracture": 0.01,
    "fracture_dislocation": 0.19,
    "greenstick_fracture": 0.4,
    "hairline_fracture": 0.08,
    "impacted_dislocation": 0.12,
    "intra-articluar_fracture": 0.07,
    "longitudinal_fracture": 0.21,
    "oblique_dislocation": 0.07,
    "pathological_fracture": 0.28,
    "spiral_fracture": 0.36
}
```