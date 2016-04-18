# Load libraries
using ODE
using DataFrames
using Gadfly
using ForwardDiff

function sigmoid(x)
  return(1 ./ (1 + exp(x)))
end

#### CONSTANT PARAMETERS #####
c = [15 -15; 15 -3]
a = [1,2]
r = [1,1]
k = [1,1]
theta = [2,2.5]
tau = [.01,.01]

function corrected_sigmoid(x)

  correction = sigmoid(a .* theta)
  return(sigmoid(-a .* (x - theta)) - correction)
end

function heaviside(t, args)
  intensity = args[1]
  duration = args[2]
  return(intensity * ((sign(duration - t) + 1) ./ 2))
end

function wcowan(t, x)
  A=x[1:2]
  input_args=x[3:4]
  P = heaviside(t, input_args)
  dA= (-A + (k - r .* A) .* corrected_sigmoid(c * A + P) )./tau
  return([dA;0;0])
end

function J_wcowan(t, x)
  input_args = x[3:4]
  P = heaviside(t, input_args)
  e_pow = -a[1] * (a[1]*c[1,1] + a[2] * c[1,2] + P - theta[1])
  i_pow = -a[2] * (a[1]*c[2,1] + a[2] * c[2,2] + P - theta[2])
  e_term = (-r[1]/tau[1]) * (sigmoid(e_pow) - sigmoid(a[1]*theta[1])) - 1
  i_term = (-r[2]/tau[2]) * (sigmoid(i_pow) - sigmoid(a[2]*theta[2])) - 1
  return [e_term 0 0 0; 0 i_term 0 0; 0 0 0 0; 0 0 0 0]
end

# Initialise model
t = linspace(0,.125,1250)
inits=[0,0,1,.125]

result = ode23s(wcowan,inits,t,jacobian=J_wcowan);

df = DataFrame();
df[:t]=result[1]
df[:E]=[r[1] for r in result[2]];
df[:I]=[r[2] for r in result[2]];
df[:EI]=df[:E] - df[:I];

p=Gadfly.plot(df,x="t",y="EI",Geom.line)

