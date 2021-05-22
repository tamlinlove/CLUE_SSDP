# CLUE for Single-Stage Decision Problems
This repo contains the implementation of CLUE (Cautiously Learning with Unreliable Experts) for Single-Stage Decision Problems, from the paper Learning Who to Trust: Policy Learning in Single-Stage Decision Problems with Unreliable Expert Advice. The CLUE framework allows you to incorporate the advice of multiple, potentially unreliable expert advisors into the Single-Stage Decision Problem loop. CLUE agents can benefit from advice from reliable experts, but are robust against advice from unreliable ones.

## Requirements
This implementation requires Python 3.8.6+, and has several dependencies, listed in [requirements.txt](requirements.txt). To install all dependencies, execute the following command.
```setup
pip install -r requirements.txt
```

## Usage
A simple panel comparison experiment with default parameters can be run using
```panel
python simple_example.py
```
