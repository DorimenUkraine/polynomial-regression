
# coding: utf-8

# In[190]:

import numpy as np
from numpy.polynomial import polynomial
from bokeh.plotting import figure, output_notebook, show
output_notebook()


# In[191]:

def poly_str(coef):
    p = ""
    i = 0
    for c in coef:
        if c==0:
            i += 1
            continue;
        if c>0:
            if i>0:
                p += "+ "
        p += "{:10.3f}".format(c)
        if i>0:
            p += "x^"+str(i)+" "
        i +=1
    return p


# In[192]:

def eval_poly(x, coef):
    return polynomial.polyval(x, coef)


# In[193]:

def poly(x_array, coef):
    return np.apply_along_axis(lambda x: eval_poly(x, coef), 0, x_array)


# In[194]:

x = np.arange(100)-50
roots = np.random.randn(np.random.randint(1, high=4)) #polynomial's roots
coef = polynomial.polyfromroots(roots) #generation
coef *= np.random.randn(len(coef)) #multiply by random numbers to randomize factors (<0)
y = poly(x, coef)
lr = [1e-5, 1e-6, 2e-10] # learning rates according to polynomial degree


# In[195]:

p = figure(title="Polynomial "+poly_str(coef), x_axis_label='x', y_axis_label='y')
p.circle(x, y, legend=poly_str(coef), line_width=2)
show(p)


# In[196]:

def mse(x, y, coef):
    return np.mean((poly(x, coef) - y)**2)


# In[197]:

def gradients(x, y, coef):
    coef_estimator = []
    for i, c in enumerate(coef):
        coef_estimator.append(np.mean(x**i * (poly(x, coef) - y)))
    return np.array(coef_estimator)


# In[198]:

def regression_polynomiale(x, y, coef, lr, epsilon=1e-4):
    prev_error = 0
    while True:
        error = mse(x, y, coef)
        if abs(error - prev_error) <= epsilon:
            break;
        prev_error = error
        grads = gradients(x, y, coef)
        coef -= lr * grads
    return coef


# In[199]:

#find best factors with the good learning rate
coef_estimator = regression_polynomiale(x, y, np.zeros(len(coef)), lr[len(coef)-2])
y_estimator = poly(x, coef_estimator)


# In[200]:

p = figure(title="Polynomial "+poly_str(coef), x_axis_label='x', y_axis_label='y')
p.circle(x, y, legend=poly_str(coef), line_width=2)
p.line(x, y_estimator, legend=poly_str(coef_estimator), line_width=2, color="orange")
show(p)


# In[ ]:




# In[ ]:



