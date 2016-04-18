using ODE
using PyPlot

const ms = .001

function sigmoid(x)
    1. ./ (1. + exp(-x))
end

function sigmoid_centered(x, a, theta)
    sigmoid(a .* (x - theta)) - sigmoid(-a .* theta)
end

# Model parameters
const c = [15 -15; 15 -3]
const tau = 10ms
const a = [1, 2]
const theta = [2, 2.5]
const r = [.8, .8]

# Input parameters
stim_duration = 1500ms
stim_intensity = .15
P(t) = t < stim_duration ? stim_intensity * ones(2) : zeros(2)

function d_activation(t, A)
    (-A + (1. + r .* A) .* sigmoid_centered(c * A + P(t), a, theta)) / tau 
end

# Initial parameters
A0 = zeros(2)
time = 0:0.01ms:1500ms

t, A = ode78(d_activation, A0, time)
A = hcat(A...)
plot(t, squeeze(A[1,:] - A[2,:], 1))
