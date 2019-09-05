using DiffEqOperators

s = x, y, z = (-5:0.2:5, -5:0.2:5, -5:0.2:5)
dx = dy = dz = x[2] - x[1]

Dxx = CenteredDifference{1}(2, 4, dx, length(x))
Dyy = CenteredDifference{2}(2, 4, dy, length(y))
Dzz = CenteredDifference{3}(2, 4, dz, length(z))
Q = compose(PeriodicBC(Float64,length.(s))...)

∇² = Dxx + Dyy + Dzz

gaussian(x::T, y::T, z::T) where T = exp(-(x^2 + y^2+ z^2))
ricker(x::T, y::T, z::T) where T = (4*(x^2+y^2+z^2) - 6)*exp(-(x^2+y^2+z^2))
u0 = [gaussian(X, Y, Z) for Z in z, Y in y, X in x]
u0xx = [ricker(X, Y, Z) for Z in z, Y in y, X in x]
A = (∇² * Q)
u_prime = A * u0


@test isapprox(u_prime, u0xx, atol=1e-3)
