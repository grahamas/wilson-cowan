module WC72

using Equations; eqn = Equations
using Plots; gr()

function calc_EI_diff(A)
    return A[:,1] - A[:,2]
end

function plot_EI_diff(t, A)
    EI_diff = calc_EI_diff(A)
    plot(t, EI_diff)
    title!("Difference between E and I")
    xaxis!("t")
    yaxis!("E - I")
end


function make_param_cell(params)
    fixed_keys = [:k,:r,:c,:alpha,:tau]
    n_fixed = length(fixed_keys)
    n_generated = 2
    param_cell = cell(n_fixed+n_generated)
    for (i,key) in enumerate(fixed_keys)
        param_cell[i] = params[key]
    end
    param_cell[n_fixed+1] = eqn.sigmoid_centered(params[:a], params[:theta])
    param_cell[n_fixed+2] = eqn.step_down(params[:input_dur], params[:input_amp])
    return param_cell
end

function diff_eq(k,r,c,alpha,tau,S,P)
    function (t, A)
        (-(alpha .* A) + (k - r .* A) .* S(c * A + P(t))) ./ tau 
    end
end

function run(param_dict, solver)
    A0 = param_dict[:A0]
    time = 0:param_dict[:simulation_dt]:param_dict[:simulation_duration]
    param_cell = make_param_cell(param_dict)
    F = diff_eq(param_cell...)
    t, A = solver(F, A0, time)
    A = hcat(A...)
    return (t, A.')
end


end

