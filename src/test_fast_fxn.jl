# compares and contrast performance improvements for different implementations. 


include("abstract_fxns.jl")
include("concrete_fxns.jl")
include("algorithm_tools.jl")
include("algorithms.jl")

using Plots, Test, LinearAlgebra

GC.enable_logging(false)

@testset "Fast Functions Interface and Implementations" begin 
    global N = 1024
    global A = randn(N, N)
    global b = randn(N)
    global f1 = ImplicitAffineNormedSquared(x -> A*x, y -> A'*y, b)
    global f2 = FastImplicitAffineNormedSquared(A, b)
    x0 = randn(N)
    g = ZeroFunction()
    max_itr = 2^16
    tol = 2^(-20)

    function SanityFISTAAmijoSlowImplementations()
        @info "RESULT 1 | FISTA  | Armijo LS | Slow Implementations "
        s = AlgoSettings(line_search_strategy=1)
        @time global RESULTS1 = fista_sanity_check(
            f1, g, x0, max_itr=max_itr, tol=tol, alg_settings=s
        )
        # RESULT 1
        global plt1 = plot(
            1:(RESULTS1|>fxn_values|>length),
            RESULTS1|>fxn_values.|>(x -> max(1e-307, x)|> log2), 
            title="Fxn", 
            label="RESULTS 1", 
            dpi=400
        )
        global plt2 = plot(
            1:(RESULTS1|>lipschitz_estimates|>length),
            RESULTS1|>lipschitz_estimates.|>(x -> max(1e-307, x)|> log2), 
            title="Lip", label="RESULTS 1",
            dpi=400
        )
        global plt3 = plot(
            1:(RESULTS1|>gradmap_values|>length),
            RESULTS1|>gradmap_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 1",
            title="Normed Gradieng Mapping", 
            dpi=400
        )
        return true
    end

    function SanityFISTAAmijoFastImplementations()
        @info "RESULT 2 | FISTA  | Armijo LS | Fast Implementations "
        s = AlgoSettings(line_search_strategy=1)
        @time global RESULTS2 = fista_sanity_check(
            f2, g, x0, max_itr=max_itr, tol=tol, alg_settings=s
        )
        # RESULT 1
        plot!(
            plt1,
            1:(RESULTS2|>fxn_values|>length),
            RESULTS2|>fxn_values.|>(x -> max(1e-307, x)|> log2), 
            title="Fxn", 
            label="RESULTS 2", 
            dpi=400
        )
        plot!(
            plt2,
            1:(RESULTS2|>lipschitz_estimates|>length),
            RESULTS2|>lipschitz_estimates.|>(x -> max(1e-307, x)|> log2), 
            title="Lip", label="RESULTS 2",
            dpi=400
        )
        plot!(
            plt3, 
            1:(RESULTS2|>gradmap_values|>length),
            RESULTS2|>gradmap_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 2",
            title="Normed Gradieng Mapping", 
            dpi=400
        )
        return true
    end

    function SanityFISTABTFastImplementations()
        @info "RESULT 1 | FISTA  | BT | Fast Implementations "
    end
    
    @test SanityFISTAAmijoSlowImplementations()
    @test SanityFISTAAmijoFastImplementations()

end


plt1 |> display
plt2 |> display
plt3 |> display