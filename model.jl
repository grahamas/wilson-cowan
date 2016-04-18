using ODE
using PyPlot

const ms = .001

function sigmoid(x)
    1. ./ (1. + exp(-x))
end

macro sigmoid_centered(a, theta)
    @eval offset = sigmoid(-$a .* $theta)
    return :(x -> sigmoid($a .* (x - $theta)) - $offset)
end

macro step_down(threshold, magnitude)
    :(t -> t < $threshold ? $magnitude * $(ones(2)) : $(zeros(2)))
end

macro diff_eq(c, tau, r, k, S, P)
    quote
        function d_activation(t, A)
            (-A + ($k + $r .* A) .* $S($c * A + $P(t))) / $tau 
        end
    end
end

# Abstract all this for generic diff eqs later
function run_model(c, tau, r, k, a, theta, input_fn)
    F = @diff_eq(c, tau, r, k, @sigmoid_centered(a, theta), input_fn)
    t, A = ode45(F, A0, time)
    A = hcat(A...)
    return t, A
end

macro explore_models_keyargs(params)
    :([Expr(:kw,x,0) for x in @eval(keys($params))]...)
end

macro def_explore_models(params)
    @eval params = $params
    quote
        function explore_models($(@explore_models_keyargs(params)))
            $(@eval(values($params))...)
        end
    end
end


function explore_models(fps::Float;  
    plot(t, squeeze(A[1,:] - A[2,:], 1))
    title(intensity)
    sleep(.4)


# Model parameters
default_parameters = Dict(
        :c =>[15. -15.; 15. -3.],
        :tau => 10ms,
        :a => [1., 2.],
        :theta => [2., 2.5],
        :r => [1., 1.],
        :k => [.9, .9]
        )

# Input parameters
stim_duration = 125ms
stim_intensity_min = .01
stim_intensity_step = .01
stim_intensity_max = 2
#P(t) = t < stim_duration ? stim_intensity * ones(2) : zeros(2)

# Initial parameters
A0 = zeros(2)
time = 0:0.01ms:125ms
intensities = 1# stim_intensity_min:stim_intensity_step:stim_intensity_max

