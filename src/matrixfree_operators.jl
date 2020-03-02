mutable struct MatrixFreeOperator{F,N,S,O} <: AbstractMatrixFreeOperator{F}
  f::F
  args::N
  size::S
  opnorm::O
  ishermitian::Bool
  function MatrixFreeOperator(f::F, args::N;
                              size=nothing, opnorm=true, ishermitian=false) where {F,N}
    @assert (N <: Tuple && length(args) in (1,2)) "Arguments of a "*
    "MatrixFreeOperator must be a tuple with one or two elements"
    return new{F,N,typeof(size),typeof(opnorm)}(f, args, size, opnorm, ishermitian)
  end
end
MatrixFreeOperator(f) = MatrixFreeOperator(f, (nothing,))

# Overloading Base functions
function Base.size(M::MatrixFreeOperator)
  M.size === nothing && error("M.size is nothing, please define size as a tuple of integers")
  return M.size
end
@inline function Base.size(M::MatrixFreeOperator, n)
  M.size === nothing && error("M.size is nothing, please define size as a tuple of integers")
  n <= 0 && error("dimension out of range")
  return n <= length(M.size) ? M.size[n] : 1
end
@inline ==(M1::MatrixFreeOperator, M2::MatrixFreeOperator) = M1.f == M2.f && M1.args == M2.args && M1.size == M2.size && M1.opnorm == M2.opnorm && M1.ishermitian == M2.ishermitian
@inline Base.:*(A::MatrixFreeOperator, X::AbstractVecOrMat) = mul!(similar(X), A, X)

# Overloading LinearAlgebra functions
LinearAlgebra.ishermitian(M::MatrixFreeOperator) = M.ishermitian
function LinearAlgebra.opnorm(M::MatrixFreeOperator, p::Real)
  M.opnorm === nothing && error("""
    M.opnorm is nothing, please define opnorm as a function that takes one
    argument. E.g. `(p::Real) -> p == Inf ? 100 : error("only Inf norm is
    defined")`
  """)
  opn = M.opnorm
  return opn isa Number ? opn : M.opnorm(p)
end

# DiffEqBase
DiffEqBase.numargs(::MatrixFreeOperator) = 4

# Interface
isconstant(M::MatrixFreeOperator) = length(M.args) == 1
function update_coefficients!(M::MatrixFreeOperator, u, p, t)
  !isconstant(M) && (M.args = (p, t))
  return M
end

function (M::MatrixFreeOperator{F,N})(du, u, p, t) where {F,N}
  update_coefficients!(M,u,p,t)
  if isconstant(M)
    M.f(du, u, p)
  else
    M.f(du, u, p, t)
  end
  du
end

function (M::MatrixFreeOperator{F,N})(u, p, t) where {F,N}
  update_coefficients!(M,u,p,t)
  du = similar(u)
  if isconstant(M)
    M.f(du, u, p)
  else
    M.f(du, u, p, t)
  end
  du
end

@inline function mul!(y::AbstractVector, A::MatrixFreeOperator{F,N}, x::AbstractVector) where {F,N}
  if isconstant(A)
    A.f(y, x, A.args[1])
  else
    A.f(y, x, A.args[1], A.args[2])
  end
  return y
end

@inline function mul!(Y::AbstractMatrix, A::MatrixFreeOperator{F,N}, X::AbstractMatrix) where {F,N}
  m,  n = size(Y)
  k, _n = size(X)
  if n != _n
    throw(DimensionMismatch("the second dimension of Y, $N, does not match the second dimension of X, $_n"))
  end
  for i in 1:n
    y = view(Y, :, i)
    x = view(X, :, i)
    mul!(y, A, x)
  end
  Y
end
