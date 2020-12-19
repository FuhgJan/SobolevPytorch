# SobolevPytorch
An implementation of a neural network training routine using available derivative information in Pytorch.

Original paper:

Czarnecki, W. M., Osindero, S., Jaderberg, M., Swirszcz, G., & Pascanu, R. (2017). Sobolev training for neural networks. In Advances in Neural Information Processing Systems (pp. 4278-4287).

Using Sobolev training we can efficiently reduce the overall loss and are able to get better approximations of the derivatives of the inputs.

## Tested on
<ul>
<li>Python 3.8</li>
<li>Numpy 1.19.4</li>
<li>Pytorch 1.7.0</li>
<li>Matplotlib 3.1.2</li>
</ul> 



## Example
Test on Franke's function

<img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;0.75&space;\exp{\left(-\frac{(9x-2)^2)}{4}&space;-&space;\frac{(9y-2)^2}{4}\right)}&space;&plus;&space;0.75&space;\exp{\left(-\frac{(9x&plus;1)^2)}{49}&space;-&space;\frac{(9y&plus;1)}{10}\right)}&space;&plus;&space;0.5&space;\exp{\left(-\frac{(9x-7)^2)}{4}&space;-&space;\frac{(9y-3)}{4}\right)}&space;-&space;0.2&space;\exp{\left(-(&space;9&space;x&space;-4)^2&space;-&space;(9y-7)^2&space;\right&space;)}" title="f(x) = 0.75 \exp{\left(-\frac{(9x-2)^2)}{4} - \frac{(9y-2)^2}{4}\right)} + 0.75 \exp{\left(-\frac{(9x+1)^2)}{49} - \frac{(9y+1)}{10}\right)} + 0.5 \exp{\left(-\frac{(9x-7)^2)}{4} - \frac{(9y-3)}{4}\right)} - 0.2 \exp{\left(-( 9 x -4)^2 - (9y-7)^2 \right )}" />

Training on 100 equidistant points between 0 and 1 yields the following convergence behavior: 

<p align="center">
<img align="middle" src="src/figures/Loss_Franke.png" alt="Normalized convergence plot" width="450" height="400" />
</p>

Testing on 1600 test points in the parametric space yields the following visualized results which are in good accordance with the target.

<p align="center">
<img align="middle" src="src/figures/PredictionOverTestPoints_Franke.png" alt="Normalized convergence plot" width="700" height="500" />
</p>
