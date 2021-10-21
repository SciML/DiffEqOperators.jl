using Documenter, DiffEqOperators

makedocs(
    sitename="DiffEqOperators.jl",
    authors="Chris Rackauckas et al.",
    clean=true,
    doctest=false,
    modules=[DiffEqOperators],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://diffeqoperators.sciml.ai/stable/"),

    pages=[
        "DiffEqOperators.jl: Linear operators for Scientific Machine Learning" => "index.md",
        "Symbolic Tutorials" => [
            "symbolic_tutorials/mol_heat.md"
        ],
        "Operator Tutorials" => [
            "operator_tutorials/kdv.md"
        ],
        "Operators" => [
            "operators/operator_overview.md",
            "operators/derivative_operators.md",
            "operators/vector_calculus_operators.md",
            "operators/vector_jacobian_product.md",
            "operators/jacobian_vector_product.md",
            "operators/matrix_free_operators.md"
        ],
        "Symbolic PDE Solver" => [
            "symbolic/molfinitedifference.md"
        ],
        "Nonlinear Derivatives" => [
            "nonlinear_derivatives/nonlinear_diffusion.md"
        ]  
     ]
)

deploydocs(
    repo="github.com/SciML/DiffEqOperators.jl";
    push_preview=true
)
