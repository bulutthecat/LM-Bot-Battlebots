# LM-Bot-Battlebots
An LM solution to the common CompSci Battlebots challange, built using a mixture of Java and Python

To get started, setup a virtual environment (reccomended to be named `venv`):
```python3 -m venv venv```
Next, install requirements:
```./venv/bin/pip3 install -r requirements.txt```
And finally the model training script:
```./venv/bin/python3 botv2.py```

Training can only be run on Linux systems with CUDA drivers and a compatible CUDA GPU.
If required, this can be bypassed by adding the `device='cpu'` flag in the botv2.py 

This is tested and confirmed functional in Python 3.10, JVM 16.