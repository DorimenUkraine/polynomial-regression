# polynomial-regression

### Please use the link provided in github to view the file with the external viewer *nbviewer*

This script generates a polynomial function : <img src="https://latex.codecogs.com/gif.latex?a_0&space;&plus;&space;a_1x&space;&plus;&space;a_2x^2&space;&plus;&space;a_3x^3" title="a_0 + a_1x + a_2x^2 + a_3x^3" />

The purpose is to find the multiplying factors that best fit the generated points.

Here the *Mean Square Error* is used and we try to minimize it with the gradient descent that figures out the gradients of our coefficients. To best fit the points, we need to use different learning rates according to the polynomial degree.

![Result](https://raw.githubusercontent.com/cheillanju/polynomial-regression/master/result.png)
