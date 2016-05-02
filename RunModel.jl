module RunModel
export run_model

using ODE
using Parameters

function run_model(model, param_file::AbstractString, solver=ode45; custom...) 
    # Bastardized OOP?
    if param_file != ""
        param_dict = Parameters.read_parameters(param_file)
    end
    return run_model(model, param_dict, solver; custom...)
end

function run_model(model, param_dict::Dict, solver=ode45; custom...)
    for (key, value) in custom
        println("$key => $value")
        param_dict[key] = value
    end
    return model.run(param_dict, solver)
end

end

