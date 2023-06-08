using Test
using DiffEqOperators
using Aqua

@testset "Aqua tests (performance)" begin
    # This tests that we don't accidentally run into
    # https://github.com/JuliaLang/julia/issues/29393
    # Aqua.test_unbound_args(DiffEqOperators) # fails
    ua = Aqua.detect_unbound_args_recursively(DiffEqOperators)
    @warn "Number of method ambiguities: $(length(ua))"
    @test length(ua) ≤ 25
    # Uncomment for debugging:
    # @show ua

    # See: https://github.com/SciML/OrdinaryDiffEq.jl/issues/1750
    # Test that we're not introducing method ambiguities across deps
    ambs = Aqua.detect_ambiguities(DiffEqOperators; recursive = true)
    pkg_match(pkgname, pkdir::Nothing) = false
    pkg_match(pkgname, pkdir::AbstractString) = occursin(pkgname, pkdir)
    filter!(x -> pkg_match("DiffEqOperators", pkgdir(last(x).module)), ambs)

    # Uncomment for debugging:
    # for method_ambiguity in ambs
    #     @show method_ambiguity
    # end
    @warn "Number of method ambiguities: $(length(ambs))"
    @test length(ambs) ≤ 30
end

@testset "Aqua tests (additional)" begin
    Aqua.test_undefined_exports(DiffEqOperators)
    # Aqua.test_stale_deps(DiffEqOperators) # fails
    Aqua.test_deps_compat(DiffEqOperators)
    Aqua.test_project_extras(DiffEqOperators)
    Aqua.test_project_toml_formatting(DiffEqOperators)
    # Aqua.test_piracy(DiffEqOperators) # failing
end

nothing
