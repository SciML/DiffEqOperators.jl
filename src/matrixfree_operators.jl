import LinearAlgebra: mul!
struct MatrixFreeOperator{F,N} <: AbstractMatrixFreeOperator{F}
  f::F
  args::N
end
MatrixFreeOperator(f) = MatrixFreeOperator(f, nothing)

@inline function mul!(y::AbstractVector, A::MatrixFreeOperator{F,N}, x::AbstractVector) where {F,N}
  if N === Nothing
    A.f(y, x)
  else
    A.f(y, x, A.args...)
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
