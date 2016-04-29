module RunModel
using ODE
using PyPlot

const ms = .001

function run_model(diff_eq, A0, time, solver=ode45)
    t, A = solver(diff_eq, A0, time)
    A = hcat(A...) # Why doesn't ode do this naturally?
    return t, A
end

A0 = zeros(2)
time = 0:0.01ms:125ms


end
