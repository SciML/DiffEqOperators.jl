struct BoundaryConditionError <: Exception
    msg::String
end

function Base.showerror(io::IO, e::BoundaryConditionError)
    print(io, "BoundaryConditionError: ", e.msg)
end
