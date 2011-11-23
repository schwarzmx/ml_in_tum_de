#!/usr/bin/env python

"""
__author__: Fernando Hernandez
"""

import scipy as sp
import numpy as np
from matplotlib.pyplot import *

def uniform(num_samples):
  return np.random.uniform(0,1,num_samples)

def gaussian(num_samples):
  return np.random.normal(0.,1.,num_samples)

def f_of_ex(x_samples, original=True, noisy=False):
  sine = []
  for x in x_samples:
    if not original:
      x = uniform(1)
    numerator = np.sin(12. * (x + 0.2) )
    denominator = x+0.2
    current = numerator/denominator
    if noisy:
      current = current + gaussian(1)
    sine.append(current)

  return sine

def pairs(num_samples):
  pairs = [uniform(num_samples),\
      f_of_ex(num_samples, noisy=True)]

  return pairs

def ridge_fit(theta, y):
  alpha = 0.5
  num_features = 1
  x = np.asanyarray(theta)
  y = np.asanyarray(y)
  
  w = np.linalg.solve(\
      np.dot(x.T, x) + alpha * np.eye(num_features), \
      np.dot(x.T,y))
     

  return w



figure(1)
samples = 10
t = np.linspace(0, 2*np.pi, samples)
# original function
subplot(221)
sine = f_of_ex(t)
plot(t,sine,ls='-.')

# function with noise
subplot(221)
noisy = f_of_ex(t,noisy=True)
plot(t,noisy, '.')

# regression
subplot(221)
#print noisy[1:]
values = sp.polyfit(t,noisy,3)
regression = sp.polyval(values,t)
plot(t,regression, '^-')
legend(['original','with noise', 'regression'])

t = np.linspace(0, 2*np.pi, samples)
# original function
subplot(222)
sine = f_of_ex(t)
plot(t,sine,ls='-.')

# function with noise
subplot(222)
noisy = f_of_ex(t,noisy=True)
plot(t,noisy, '.')

# regression
subplot(222)
values = sp.polyfit(t,noisy,3)
regression = sp.polyval(values,t)
plot(t,regression, '-.')
legend(['original','with noise', 'regression'])

samples = 100
t = np.linspace(0, 2*np.pi, samples)
# original function
subplot(223)
sine = f_of_ex(t)
plot(t,sine,ls='-.')

# function with noise
subplot(223)
noisy = f_of_ex(t,noisy=True)
plot(t,noisy, '.')

# regression
subplot(223)
values = sp.polyfit(t,noisy,3)
regression = sp.polyval(values,t)
plot(t,regression, 'o')

# ridge regression
subplot(223)
sqrt_alpha = np.sqrt(5.)
t = np.append(t,[sqrt_alpha])
noisy = np.append(noisy,np.zeros(1))
coeficients = sp.polyfit(t,noisy,3)
ridge = sp.polyval(coeficients,t)
plot(t,ridge, '^')
legend(['original','with noise', 'regression', 'ridge'])

t = np.linspace(0, 2*np.pi, samples)
# original function
subplot(224)
sine = f_of_ex(t)
plot(t,sine,ls='-.')

# function with noise
subplot(224)
noisy = f_of_ex(t,noisy=True)
plot(t,noisy, '.')

# regression
subplot(224)
values = sp.polyfit(t,noisy,9)
regression = sp.polyval(values,t)
plot(t,regression, 'o')

# ridge regression
subplot(224)
sqrt_alpha = np.sqrt(5.)
t = np.append(t,[sqrt_alpha])
noisy = np.append(noisy,np.zeros(1))
coeficients = sp.polyfit(t,noisy,9)
ridge = sp.polyval(coeficients,t)
plot(t,ridge, '^')
legend(['original','with noise', 'regression', 'ridge'])

show()




print "Finished!"
