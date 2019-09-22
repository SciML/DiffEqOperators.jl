using SparseArrays, DiffEqOperators, LinearAlgebra, Random,
    Test, BandedMatrices, FillArrays, LazyArrays, BlockBandedMatrices

@testset "Concretizations of BCs" begin
    T = Float64
    L = 10one(T)
    N = 9
    δx = L/(N+1)

    @testset "Affine BCs" begin
        @testset "Dirichlet0BC" begin
            Q = Dirichlet0BC(T)

            correct = vcat(zeros(T,1,N),
                           Diagonal(ones(T,N)),
                           zeros(T,1,N))

            @testset "$mode concretization" for (mode,Mat,Expected,ExpectedBandwidths) in [
                ("sparse -> Banded", sparse, BandedMatrix{T}, (1,-1)),
                ("Banded", BandedMatrix, BandedMatrix{T}, (1,-1)),
                ("Sparse", SparseMatrixCSC, SparseMatrixCSC{T}, nothing),
                ("Dense", Array, Matrix{T}, nothing)
            ]
                Qm,Qu = Mat(Q,N)

                @test Qm == correct
                @test Qm isa Expected
                @test Qu == zeros(T,N+2)

                !isnothing(ExpectedBandwidths) &&
                    @test bandwidths(Qm) == ExpectedBandwidths
            end
        end

        @testset "Neumann0BC" begin
            Q = Neumann0BC(δx)

            correct = vcat(hcat(one(T),zeros(T,1,N-1)),
                           Diagonal(ones(T,N)),
                           hcat(zeros(T,1,N-1),one(T)))

            @testset "$mode concretization" for (mode,Mat,Expected,ExpectedBandwidths) in [
                ("sparse -> Banded", sparse, BandedMatrix{T}, (2,0)),
                ("Banded", BandedMatrix, BandedMatrix{T}, (2,0)),
                ("Sparse", SparseMatrixCSC, SparseMatrixCSC{T}, nothing),
                ("Dense", Array, Matrix{T}, nothing)
            ]
                Qm,Qu = Mat(Q,N)

                @test Qm == correct
                @test Qm isa Expected
                @test Qu == zeros(T,N+2)

                !isnothing(ExpectedBandwidths) &&
                    @test bandwidths(Qm) == ExpectedBandwidths
            end

            @testset "Banded concretization, extra zeros" begin
                @testset "lz = $lz" for lz = 0:3
                    @testset "rz = $rz" for rz = 0:3
                        Q′ = Neumann0BC(δx)
                        # Artificially add some zero coefficients, which should
                        # not increase the bandwidth of the concretized BC.
                        append!(Q′.a_l, zeros(lz))
                        append!(Q′.a_r, zeros(rz))

                        Q′m,Q′u = sparse(Q′,N)
                        @test bandwidths(Q′m) == (2,0)

                        @test Q′m == correct
                        @test Q′u == zeros(T,N+2)
                    end
                end
            end
        end

        @testset "General BCs" begin
            @testset "Left BC order = $ld" for ld = 2:5
                @testset "Right BC order = $rd" for rd = 2:5
                    αl = 0.0:ld-1
                    αr = 0.0:rd-1

                    Q = GeneralBC(αl, αr, δx)

                    correct = vcat(hcat(Q.a_l',zeros(T,1,N-(ld-2))),
                                   Diagonal(ones(T,N)),
                                   hcat(zeros(T,1,N-(rd-2)),Q.a_r'))

                    Qm,Qu = sparse(Q,N)

                    @test Qm == correct
                    @test Qm isa BandedMatrix{T}
                    @test bandwidths(Qm) == (rd-1,ld-3)

                    @test Qu == vcat(Q.b_l,zeros(T,N),Q.b_r)
                end
            end
        end

        @testset "Dirichlet0BC" begin
            # This is equivalent to a Dirichlet0BC; the trailing zeros
            # should be dropped and the bandwidths optimal.
            Q = GeneralBC([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], δx)

            correct = vcat(zeros(T,1,N),
                           Diagonal(ones(T,N)),
                           zeros(T,1,N))

            Qm = first(sparse(Q,N))
            @test Qm == correct
            @test bandwidths(Qm) == (1,-1)
        end

        @testset "Almost DirichletBC" begin
            Q = GeneralBC([1.0, 1.0, 0.0, 0.0, eps(Float64)],
                          [1.0, 1.0, 0.0, 0.0, 0.0], δx)

            correct = vcat(zeros(T,1,N),
                           Diagonal(ones(T,N)),
                           zeros(T,1,N))

            Qm,Qu = sparse(Q,N)

            @test Qm ≈ correct
            @test bandwidths(Qm) == (1,2)
            @test Qu ≈ vcat(-one(T),zeros(T,N),-one(T))
        end
    end

    @testset "Periodic BCs" begin
        Q = PeriodicBC(T)
        @test_throws ArgumentError BandedMatrix(Q,N)

        correct = vcat(hcat(zeros(T,1,N-1),one(T)),
                       Diagonal(ones(T,N)),
                       hcat(one(T),zeros(T,1,N-1)))

        @testset "Sparse concretization" begin
            Qm,Qu = SparseMatrixCSC(Q,N)

            @test Qm == correct
            @test Qm isa SparseMatrixCSC{T}
            @test Qu == zeros(T,N+2)

            Qm′ = first(sparse(Q, N))
            @test Qm′ == correct
            @test Qm′ isa SparseMatrixCSC{T}
        end

        @testset "Dense concretization" begin
            Qm,Qu = Array(Q,N)

            @test Qm == correct
            @test Qm isa Matrix{T}
            @test Qu == zeros(T,N+2)
        end
    end
end
