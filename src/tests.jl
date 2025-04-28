include("abstract_fxns.jl")
include("concrete_fxns.jl")
include("algorithm_tools.jl")
include("algorithms.jl")

using Plots, Test, LinearAlgebra

@testset "Basic Testing" begin
    global N = 256^2
    global A = LinRange(0, 1, N)|>collect
    global b = zeros(N)
    global f = ENormSquaredViaLinMapImplicit(x -> A.*x, y -> A.*y, b)
    x0 = randn(N)
    g = ZeroFunction()
    max_itr=2^14
    tol=2^(-25)

    function visualize_results(c::ResultsCollector)::Nothing

        return nothing
    end
    
    function sanity_test()
        

        return true
    end 

    function basic_run_armijo()
        @info "RESULT 1 | Alamo Restart  | BT LS "
        s = AlgoSettings(restart_strategy=2, line_search_strategy=1, monotone_strategy=2)
        @time global RESULTS1 = fista(f, g, x0, max_itr=max_itr, tol=tol, alg_settings=s)
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
            title="Lip", label="FISTA | RESULTS 1",
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

    function basic_run_backtrack()
        @info "RESULT 2 | Chambolle's backtrack"
        s = AlgoSettings(line_search_strategy=1)
        @time global RESULTS2 = fista(f, g, x0, max_itr=max_itr, tol=tol, alg_settings=s)
        
        # RESULT 2
        plot!(
            plt1,
            1:(RESULTS2|>fxn_values|>length),
            RESULTS2|>fxn_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 2", 
        )
        plot!(
            plt2,
            1:(RESULTS2|>lipschitz_estimates|>length),
            RESULTS2|>lipschitz_estimates.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 2"
        )
        plot!(
            plt3,
            1:(RESULTS2|>gradmap_values|>length),
            RESULTS2|>gradmap_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 2",
            dpi=400
        )
        return true
    end

    function basic_run_armijo_beckmono()
        @info "RESULT 3 | Armijo | Bekc's Mono"
        s = AlgoSettings(line_search_strategy=0, monotone_strategy=1)
        @time global RESULTS3 = fista(f, g, x0, max_itr=max_itr, alg_settings=s, tol=tol)
        
        # RESULT 3
        plot!(
            plt1,
            1:(RESULTS3|>fxn_values|>length),
            RESULTS3|>fxn_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 3"
        )
        plot!(
            plt2,
            1:(RESULTS3|>lipschitz_estimates|>length),
            RESULTS3|>lipschitz_estimates.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 3"
        )
        plot!(
            plt3, 
            1:(RESULTS3|>gradmap_values|>length),
            RESULTS3|>gradmap_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 3",
            dpi=400
        )
        return true
    end

    function basic_run_backtrack_beckmono()
        @info "RESULT 4 | Chambolle's Backtrack | Beck's Monotone"
        s = AlgoSettings(line_search_strategy=1, monotone_strategy=1)
        @time global RESULTS4 = fista(f, g, x0, max_itr=max_itr, alg_settings=s, tol=tol)
        # RESULT 4
        plot!(
            plt1,
            1:(RESULTS4|>fxn_values|>length),
            RESULTS4|>fxn_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 4"
        )
        plot!(
            plt2,
            1:(RESULTS4|>lipschitz_estimates|>length),
            RESULTS4|>lipschitz_estimates.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 4"
        )
        plot!(
            plt3, 
            1:(RESULTS4|>gradmap_values|>length),
            RESULTS4|>gradmap_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 4",
            dpi=400
        )
        return true
    end

    function basic_run_backtrack_nesmono()
        @info "RESULT 5 | Chambolle's Backtrack | Nesterov's Monotone"
        s = AlgoSettings(line_search_strategy=1, monotone_strategy=2)
        @time global RESULTS5 = fista(f, g, x0, max_itr=max_itr, alg_settings=s, tol=tol)
        
        # RESULTS 5
        plot!(
            plt1, 
            1:(RESULTS5|>fxn_values|>length),
            RESULTS5|>fxn_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 5"
        )
        plot!(
            plt2, 
            1:(RESULTS5|>lipschitz_estimates|>length),
            RESULTS5|>lipschitz_estimates.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 5"
        )
        plot!(
            plt3,
            1:(RESULTS5|>gradmap_values|>length),
            RESULTS5|>gradmap_values.|>(x -> max(1e-307, x)|> log2), 
            label="RESULTS 5",
            dpi=400
        )
        return true
    end

    @test sanity_test()
    @test basic_run_armijo()
    @test basic_run_backtrack()
    @test basic_run_armijo_beckmono()
    @test basic_run_backtrack_beckmono()
    @test basic_run_backtrack_nesmono()
    
end




plt1|>display
plt2|>display
plt3|>display