using Documenter, DiffEqOperators

include("pages.jl")

makedocs(sitename = "DiffEqOperators.jl",
         authors = "Chris Rackauckas et al.",
         clean = true,
         doctest = false,
         modules = [DiffEqOperators],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://diffeqoperators.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/DiffEqOperators.jl";
           push_preview = true)
