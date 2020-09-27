using DiffEqOperators, DifferentialEquations, ProgressMeter

s = x, y, z = (-5:0.2:5, -5:0.2:5, -5:0.2:5)
dx = dy = dz = x[2] - x[1]

ricker(x::T, y::T, z::T) where T = (4*(x^2+y^2+z^2) - 6)*exp(-(x^2+y^2+z^2))

u0 = [ricker(X, Y, Z) for Z in z, Y in y, X in x]

Dxx = CenteredDifference{1}(2, 4, dx, length(x))
Dyy = CenteredDifference{2}(2, 4, dy, length(y))
Dzz = CenteredDifference{3}(2, 4, dz, length(z))

A = Dxx+Dyy+Dzz
Q = compose(Dirichlet0BC(Float64, length.(s))...)

dt = dx/(sqrt(3)*3e8)
t = 0.0:dt:10/3e8

f(u,p,t) = (3e8)^2 .*(A*Q*u)



function steptime(u,uold,uolder)
    return ((dt^2 .*f(u,0,0) .+ 5u .- 4uold .+ uolder)./2, u, uold)
end
let uolder = deepcopy(u0), uold = deepcopy(u0), u = deepcopy(u0)
    u,uold,uolder = steptime(u0,uold,uolder)
    # @gif for ti in t #4th order time stepper
    #         u, uold, uolder = steptime(u,uold,uolder)
    #         heatmap(u)
    # end
    for ti in t #4th order time stepper
        u, uold, uolder = steptime(u,uold,uolder)
    end
end

