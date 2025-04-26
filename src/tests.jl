include("abstract_fxns.jl")
include("concrete_fxns.jl")
include("algorithm_tools.jl")
include("algorithms.jl")

using UnicodePlots, Test, LinearAlgebra

@testset "Basic Testing" begin
    global N = 3
    global A = Diagonal(LinRange(0, 1, N))
    global f = ENormSquaredViaLinMapImplicit((x) ->(A*x), (y) -> (A'y), zeros(N))
    x0 = ones(N)
    g = ZeroFunction()
    
    function sanity_test()
        

        return true
    end 

    function basic_run_armijo()
        x = fista(f, g, x0, max_itr=100)
        display(x)
        return true
    end

    @test sanity_test()
    @test basic_run_armijo()


end