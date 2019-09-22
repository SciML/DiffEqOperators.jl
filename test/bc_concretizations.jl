using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
    Test, BandedMatrices, FillArrays, LazyArrays, BlockBandedMatrices

@testset "Concretizations of BCs" begin
    @testset "Periodic BCs" begin
        N = 9
        T = Float64
        Q = PeriodicBC(T)
        @test_throws ArgumentError BandedMatrix(Q,N)

        correct = vcat(hcat(zeros(T,1,N-1),one(T)),
                       Diagonal(ones(T,N)),
                       hcat(one(T),zeros(T,1,N-1)))

        @testset "Sparse concretization" begin
            Qm,Qu = SparseMatrixCSC(Q,N)

            @test Qm == correct
            @test Qm isa SparseMatrixCSC{T}
            @test Qu == zeros(T,N)
        end

        @testset "Dense concretization" begin
            Qm,Qu = Array(Q,N)

            @test Qm == correct
            @test Qm isa Matrix{T}
            @test Qu == zeros(T,N)
        end
    end
end
