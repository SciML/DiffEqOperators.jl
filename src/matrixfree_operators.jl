import LinearAlgebra: mul!
struct MatrixFreeOperator{F,N} <: AbstractMatrixFreeOperator{F}
  f::F
  args::N
  function MatrixFreeOperator(f::F, args::N) where {F,N}
    @assert N === Nothing || (N <: Tuple && length(args) == 2) "Arguments of a "*
    "MatrixFreeOperator must be nothing or a tuple with two elements"
    return new{F,N}(f, args)
  end
end
MatrixFreeOperator(f) = MatrixFreeOperator(f, nothing)

function Base.getproperty(M::MatrixFreeOperator{F,N}, sym::Symbol) where {F,N}
  if sym === :update_func
    return N === Nothing ? DEFAULT_UPDATE_FUNC : M.f
  else
    return getfield(M, sym)
  end
end

function (M::MatrixFreeOperator{F,N})(du, u, p, t) where {F,N}
  if N === Nothing
    M.f(du, u)
  else
    M.f(du, u, p, t)
  end
  du
end

function (M::MatrixFreeOperator{F,N})(u, p, t) where {F,N}
  du = similar(u)
  if N === nothing
    M.f(du, u)
  else
    M.f(du, u, p, t)
  end
  du
end

@inline function mul!(y::AbstractVector, A::MatrixFreeOperator{F,N}, x::AbstractVector) where {F,N}
  if N === Nothing
    A.f(y, x)
  else
    A.f(y, x, A.args[1], A.args[2])
  end
  y
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
@inline Base.:*(A::MatrixFreeOperator, X::AbstractVecOrMat) = mul!(similar(X), A, X)

DiffEqBase.numargs(::MatrixFreeOperator) = 4
