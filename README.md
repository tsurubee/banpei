## About Banpei
Banpei is a Python package of the anomaly detection.  
Anomaly detection is a technique used to identify unusual patterns that do not conform to expected behavior.

## System
Python 3.x (2.x is not supported)

## Installation
Use the command:
```bash
$ git clone https://github.com/tsurubee/banpei.git
$ cd banpei
$ pip install banpei
```
After installation, you can import banpei in Python.
```
$ python
>>> import banpei
```

## Usage
#### Example
*Singular spectrum transformation(sst)*
```python
import banpei 
model   = banpei.SST(data, w=50)
results = model.detect()
```
The graph below shows the change-point scoring calculated by sst for the periodic data. (The data used is placed as '/tests/test_data/periodic_wave.csv')

<img src="./docs/images/sst_example.png" alt="sst_example" width="450">

## The implemented algorithm
#### Outlier detection
* Hotelling's theory
#### Change point detection
* Singular spectrum transformation(sst)

## License
This project is licensed under the terms of the MIT license, see LICENSE.