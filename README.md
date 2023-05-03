# Crafter
CRAFT text detection with ONNX Runtime

Based on the [craft-text-detector](https://github.com/innodatalabs/craft-text-detector). See also the source of the fork [here](https://github.com/fcakyon/craft-text-detector).

## Installation
```bash
$ pip install crafter
```

## Usage
```python
from crafter import Crafter

crafter = Crafter()

prediction = crafter('crafter/test/resources/idcard2.jpg')
for p1, p2, p3, p4 in prediction['boxes']:
    print(p1, p2, p3, p4)
```

## Developing
```bash
$ pip install .
$ pip install onnx git@github.com:innodatalabs/craft-text-detector.git pytest
```

To download Pytorch weights and convert to ONNX, run this (once):
```bash
$ python convert/craftnet.py
$ python convert/refinenet.py
```
This will (re-)create the ONNX files in `crafter/resources`.

## Testing
```bash
$ PYTHONPATH+. pytest
```

## Building
```bash
$ make
```