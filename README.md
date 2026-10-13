Code for the paper: Learning Gradual Argumentation for Decision Making

# Execution

To execute the experimentation run:

```
python3 train.py
```


# Setup

## Virtual environment

This code was tested on a virtual environment (venv) on python 3.11 (due to Uncertainpy requirements)

## Library adaptation

Important: the uncertainpy code (in local) was modified for this experimentation:

./Uncertainpy/src/gradual/BAG.py

`New file can be found in BAG.py in the root of this repo, it will work if you copy that file into the package`



./Uncertainpy/src/gradual/semantics/modular/MLPBasedInfluence.py

'''
class MLPBasedInfluence:
    def __init__(self) -> None:
        pass

    def compute_strength(self, weight, aggregate):
        
        if aggregate == 0:
            return weight
        
        if 1-weight == 0:
            return 1
        elif weight == 0:
            return 0
        
        return 1/(1 + math.exp(- math.log(weight/(1-weight)) - aggregate))

    def __str__(self) -> str:
        return __class__.__name__ 
'''
