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
        "Examples" => "examples.md"
    ]
)

deploydocs(
    repo="github.com/SciML/DiffEqOperators.jl";
    push_preview=true
)