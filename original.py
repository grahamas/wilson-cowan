import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import ode

#tau = T.vector('tau')
#r, k = T.scalars('r', 'k')
#c = T.matrix('c')
#a = T.vector('a')

c = np.array([[15, -15], [15, -3]])
a = np.array([1, 2])
r = np.array([1, 1])
theta = np.array([2, 2.5])
tau = np.array([.01, .01])

activation = T.vector('activation')
t = T.scalar('t')
input_duration = T.scalar('input_duration')
input_intensity = T.scalar('input_intensity')

# TODO: Define input function P, heaviside
#P = theano.function([t, input_duration, input_intensity],input_intensity * ((T.sgn(input_duration - t) + 1) / 2))
corrected_sigmoid = 1 / (1 + T.exp(T.mul(a,((T.dot(c,activation) + input_intensity * ((T.sgn(input_duration - t) + 1) / 2)) - theta)))) - 1 / (1 + T.exp(np.multiply(a, theta)))
#corrected_sigmoid = theano.function([t, activation, input_duration, input_intensity], corrected_s)
#d_a = T.true_div(-activation + T.mul( 1 - T.mul(r, activation), corrected_s), tau)
d_a = T.true_div(-activation + T.mul( 1 - T.mul(r, activation), corrected_sigmoid), tau)
d_activation = theano.function(inputs=[t, activation, input_duration, input_intensity], outputs=d_a)

activation_0 = np.array([0, 0])
t_0 = 0

intens = 1
duration = .025

r = ode(d_activation).set_integrator('vode')
r.set_initial_value(activation_0).set_f_params(intens, duration)

theano.pp(d_activation)
t1 = .125
dt = .0001

times = []
timeseries = []

while r.successful() and r.t <= t1:
    print(r.t)
    times.append(r.t)
    timeseries.append(r.integrate(r.t+dt))

E_series = [out[0] for out in timeseries]
I_series = [out[1] for out in timeseries]

plt(time, np.array(E_series) - np.array(I_series))

