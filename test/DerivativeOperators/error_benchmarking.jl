using DiffEqOperators
# using Plots

n = 10000
@show n
x = range(0.0; length = n, stop = 2π)
dx = x[2] - x[1]
y = exp.(π * x)
y_ = y[2:(end-1)]

for dor = 1:4, aor = 2:6

    D1 = CenteredDifference(dor, aor, dx, length(x))

    #test result
    @show dor
    @show aor
    #take derivatives
    err_abs = abs.(D1 * y .- (π^dor) * y_)
    err_percent = 100 * err_abs ./ abs.(((π^dor) * y_))
    max_error = maximum(err_percent) # test operator with known derivative of exp(kx)
    avg_error = sum(err_percent) / length(err_percent)

    @show max_error
    @show avg_error
    #plot(x[2:(end-1)], err_percent, title="Percentage Error, n=$n aor = $aor, dor = $dor")

    #test result
end
