struct BoundaryConditionError <: Exception
    msg::String
end

Base.showerror(io::IO, e::BoundaryConditionError) = print(
    io, "BoundaryConditionError: ", e.msg
)