# Functional Optimal Transport

<img src="https://raw.githubusercontent.com/VersElectronics/FOT/master/files/swiss_roll.gif" height="300" width="1000">

This repository contains the codebase accompaning the 
paper "Functional Optimal Transport: Mapping Estimation and Domain Adaptation for Functional data".


## Getting Started

#### Basic requirements

```
python 3
numpy
matplotlib
scipy
```
#### Installation
```angular2html
git clone https://github.com/VersElectronics/FOT.git
```


## Usage 


#### File description
* `fot/FOT_Solver.py`   The FOT solver class
* `toy_example/`    Toy examples.


### Run toy examples

```python
python3 FOT_toy_example_03.py
```
<img src="https://raw.githubusercontent.com/VersElectronics/FOT/master/files/toy_example_03.png" height="150" width="600">

[comment]: <> (### Robot-arm)

[comment]: <> (#### Comparison)

[comment]: <> (#### Advanced Usage)

[comment]: <> (Start by creating an FOT solver class and input the data.)

[comment]: <> (```python)

[comment]: <> (from FOT import xxx)

[comment]: <> (```)

[comment]: <> (Set the initial values)

[comment]: <> (```python)

[comment]: <> (GFOT_optimizer.Set_Initial_Variables&#40;ini_A=ini_A, ini_Pi=ini_Pi,)

[comment]: <> (                                         ini_lbd_k=lbd_k, ini_lbd_l=lbd_l,)

[comment]: <> (                                         ini_lbd_i=lbd_i, s_mat=s_mat&#41;)

[comment]: <> (```)

[comment]: <> (Set the parameters for )


## Acknowledgements

This repo adopted the following packages as benchmarks

* https://github.com/vivienseguy/Large-Scale-OT

* https://github.com/VersElectronics/WGPOT
