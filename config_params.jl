module Parameters
export read_parameters

using JSON

function process_parameter(param, tp=Float64, depth=1)
    if typeof(param) <: Array
        return cat(depth, [process_param(p,tp,depth+1) for p in param]...)
    else
        return tp(param)
    end
end

function parse_keyvalue(kv)
    return (kv[1], process_parameter(kv[2]))
end

function read_parameters(file_name, tp=Float64)
    raw_dct = JSON.parsefile(file_name)
    dct = map(parse_keyvalue, raw_dct)
    return dct
end

end
