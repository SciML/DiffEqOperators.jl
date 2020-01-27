struct MOLFiniteDifference{T} <: DiffEqBase.AbstractDiscretization
  dxs::T
  order::Int
end
MOLFiniteDifference(args...;order=2) = MOLFiniteDifference(args,order)

function DiffEqBase.discretize(pdesys::PDESystem,discretization::MOLFiniteDifference)
  tdomain = pdesys.domain[1].domain
  domain = pdesys.domain[2].domain
  @assert domain isa IntervalDomain
  len = domain.upper - domain.lower
  dx = discretization.dxs[1]
  interior = domain.lower+dx:dx:domain.upper-dx
  X = domain.lower:dx:domain.upper
  L = CenteredDifference(2,2,dx,Int(len/dx)-2)
  Q = DirichletBC(0.0,0.0)
  function f(du,u,p,t)
    mul!(du,L,Array(Q*u))
  end
  u0 = @. - interior * (interior - 1) * sin(interior)
  PDEProblem(ODEProblem(f,u0,(tdomain.lower,tdomain.upper),nothing),Q,X)
end
