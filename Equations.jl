module Equations

function sigmoid(x)
    return 1. ./ (1. + exp(-x))
end

function sigmoid_centered(a, theta)
    offset = sigmoid(-a .* theta)
    return (x) -> sigmoid(a .* (x - theta)) - offset
end

function step_down(duration, amplitude)
    return t -> t .< duration ? amplitude * ones(2) : zeros(2)
end

end
