using DiffEqOperators

dx = rand(100)/10;
x = [0.0;cumsum(dx)]

D = DiffEqOperators.FiniteDifference{Float64}(1,2,dx,length(x),:None,:None)

full(D)
