using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

#Start Test Script

@time begin
if GROUP == "All" || GROUP == "Interface"
    @time @safetestset "Utilities Tests" begin include("utils.jl") end
    @time @safetestset "Poisson example" begin include("../examples/poisson.jl") end
    @time @safetestset "Heat equation example" begin include("../examples/heat_equation.jl") end
    @time @safetestset "Robin Boundary Condition Operators" begin include("robin.jl") end
    @time @safetestset "JacVec Operators Interface" begin include("jacvec_operators.jl") end
    @time @safetestset "Composite Operators Interface" begin include("composite_operators_interface.jl") end
    @time @safetestset "BC and Coefficient Compositions" begin include("bc_coeff_compositions.jl") end
    @time @safetestset "Derivative Operators Interface" begin include("derivative_operators_interface.jl") end
    @time @safetestset "Ghost Derivative Operators Interface" begin include("ghost_derivative_operators_interface.jl") end
    @time @safetestset "Validate Regular Derivative Operators" begin include("regular_operator_validation.jl") end
    @time @safetestset "Validate and Compare Generic Operators" begin include("generic_operator_validation.jl") end
    @time @safetestset "Validate Boundary Padded Array Concretization" begin include("boundary_padded_array.jl") end
    @time @safetestset "Validate Higher Dimensional Boundary Extension" begin include("MultiDimBC_test.jl") end
    @time @safetestset "2nd order check" begin include("2nd_order_check.jl") end
    #@time @safetestset "KdV" begin include("KdV.jl") end # KdV times out and all fails
    #@time @safetestset "Heat Equation" begin include("heat_eqn.jl") end
    @time @safetestset "Matrix-Free Operators" begin include("matrixfree.jl") end
    @time @safetestset "JacVec Operator Integration Test" begin include("jacvec_integration_test.jl") end
    @time @safetestset "Convolutions" begin include("convolutions.jl") end
    @time @safetestset "Differentiation Dimension" begin include("differentiation_dimension.jl") end
    @time @safetestset "Higher Dimensional Concretization" begin include("concretization.jl") end
    @time @safetestset "Coefficient Functions" begin include("coefficient_functions.jl") end
    @time @safetestset "Upwind Operator Interface" begin include("upwind_operators_interface.jl") end
    @time @safetestset "MOLFiniteDifference Interface" begin include("MOLtest.jl") end
    @time @safetestset "Basic SDO Examples" begin include("BasicSDOExamples.jl") end
    @time @safetestset "3D laplacian Test" begin include("3D_laplacian.jl") end
    # @time @safetestset "Linear Complementarity Problem Examples" begin include("lcp.jl"); include("lcp_split.jl") end
end

if !is_APPVEYOR && (GROUP == "All" || GROUP == "Multithreading")
    @time @safetestset "2D and 3D fast multiplication" begin include("2D_3D_fast_multiplication.jl") end
end
end
