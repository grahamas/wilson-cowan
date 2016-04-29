module WC72

using Equations
import Parameters

default_parameters = Dict(
        :c =>[15. -15.; 15. -3.],
        :tau => 10ms,
        :a => [1., 2.],
        :theta => [2., 2.5],
        :r => [1., 1.],
        :k => [.9, .9],
        :alpha => [1, 1],
        :input_dur = 125ms,
        :input_amp = .75,
        :simulation_dur = 250ms,
        :dt = .01ms
        )

function process_params(params)
    processed_params = Dict()
    fixed_keys = [:c, :tau, :r, :k, :alpha]
    map((key) -> processed_params[key] = params[key])
    processed_params[:S] = sigmoid_centered(params[:a], params[:theta])
    processed_params[:P] = step_down(params[:input_dur], params[:input_amp])
    return processed_params
end

function diff_eq(params)
    k = params[:k]
    r = params[:r]
    S = params[:S]
    c = params[:c]
    P = params[:P]
    alpha = params[:alpha]
    tau = params[:tau]
    function (t, A)
        (-(alpha .* A) + (k - r .* A) .* S(c * A + P(t))) / tau 
    end
end

function EI_diff(A)
    return A[1,:] - A[2,:] # should probably transpose
end

end

