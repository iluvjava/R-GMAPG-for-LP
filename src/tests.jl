include("abstract_fxns.jl")
include("concrete_fxns.jl")
include("algorithm_tools.jl")
include("algorithms.jl")

using Plots, Test, LinearAlgebra

@testset "Basic Testing" begin
    global N = 8
    global A = Diagonal(LinRange(1e-2, 1, N))
    global f = ENormSquaredViaLinMapImplicit((x) ->(A*x), (y) -> (A'y), zeros(N))
    x0 = randn(N)
    g = ZeroFunction()
    max_itr=10000
    tol=1e-18

    function visualize_results(c::ResultsCollector)::Nothing

        return nothing
    end
    
    function sanity_test()
        

        return true
    end 

    function basic_run_armijo()
        @info "FISTA | Armjio "
        global RESULTS1 = fista(f, g, x0, max_itr=max_itr, tol=tol)
        
        return true
    end

    function basic_run_backtrack()
        @info "FISTA | Chambolle's backtrack"
        s = AlgoSettings(line_search_strategy=1)
        global RESULTS2 = fista(f, g, x0, max_itr=max_itr, tol=tol, alg_settings=s)
        
        return true
    end

    function basic_run_armijo_beckmono()
        @info "N'MFISTA | Armijo | Bekc's Mono"
        s = AlgoSettings(line_search_strategy=0, monotone_strategy=1)
        global RESULTS3 = fista(f, g, x0, max_itr=max_itr, alg_settings=s, tol=tol)
       
        return true
    end

    function basic_run_backtrack_beckmono()
        @info "MFISTA | Chambolle's Backtrack | Beck's Monotone"
        s = AlgoSettings(line_search_strategy=1, monotone_strategy=1)
        global RESULTS4 = fista(f, g, x0, max_itr=max_itr, alg_settings=s, tol=tol)
        
        return true
    end

    function basic_run_backtrack_nesmono()
        @info "MFISTA | Chambolle's Backtrack | Nesterov's Monotone"
        s = AlgoSettings(line_search_strategy=1, monotone_strategy=2)
        global RESULTS5 = fista(f, g, x0, max_itr=max_itr, alg_settings=s, tol=tol)
        
        return true
    end

    @test sanity_test()
    @test basic_run_armijo()
    @test basic_run_backtrack()
    @test basic_run_armijo_beckmono()
    @test basic_run_backtrack_beckmono()
    @test basic_run_backtrack_nesmono()
    
end

# RESULT 1
plt1 = plot(
    1:(RESULTS1|>fxn_values|>length),
    RESULTS1|>fxn_values.|>(x -> max(-53, x)|> log2), 
    title="Fxn", 
    label="Armijo LS", 
    dpi=400
)
plt2 = plot(
    1:(RESULTS1|>lipschitz_estimates|>length),
    RESULTS1|>lipschitz_estimates.|>(x -> max(-53, x)|> log2), 
    title="Lip",
    dpi=400
)
plt3 = plot(
    1:(RESULTS1|>gradmap_values|>length),
    RESULTS1|>gradmap_values.|>(x -> max(-53, x)|> log2), 
    label="Armijo LS",
    title="Normed Gradieng Mapping", 
    dpi=400
)
# RESULT 2
plot!(
    plt1,
    1:(RESULTS2|>fxn_values|>length),
    RESULTS2|>fxn_values.|>(x -> max(-53, x)|> log2), 
    label="BT", 
)
 plot!(
    plt2,
    1:(RESULTS2|>lipschitz_estimates|>length),
    RESULTS2|>lipschitz_estimates.|>(x -> max(-53, x)|> log2), 
    label="BT"
)
plot!(
    plt3,
    1:(RESULTS2|>gradmap_values|>length),
    RESULTS2|>gradmap_values.|>(x -> max(-53, x)|> log2), 
    label="BT",
    dpi=400
)
# RESULT 3
plot!(
    plt1,
    1:(RESULTS3|>fxn_values|>length),
    RESULTS3|>fxn_values.|>(x -> max(-53, x)|> log2), 
    label="B'Mono | Armijo"
)
plot!(
    plt2,
    1:(RESULTS3|>lipschitz_estimates|>length),
    RESULTS3|>lipschitz_estimates.|>(x -> max(-53, x)|> log2), 
    label="B'Mono | Armijo"
)
plot!(
    plt3, 
    1:(RESULTS3|>gradmap_values|>length),
    RESULTS3|>gradmap_values.|>(x -> max(-53, x)|> log2), 
    label="B'Mono | Armijo",
    dpi=400
)
# RESULT 4
plot!(
    plt1,
    1:(RESULTS4|>fxn_values|>length),
    RESULTS4|>fxn_values.|>(x -> max(-53, x)|> log2), 
    label="B'Mono | BT"
)
plot!(
    plt2,
    1:(RESULTS4|>lipschitz_estimates|>length),
    RESULTS4|>lipschitz_estimates.|>(x -> max(-53, x)|> log2), 
    label="B'Mono | BT"
)
plot!(
    plt3, 
    1:(RESULTS4|>gradmap_values|>length),
    RESULTS4|>gradmap_values.|>(x -> max(-53, x)|> log2), 
    label="B'Mono | BT",
    dpi=400
)
# RESULTS 5
plot!(
    plt1, 
    1:(RESULTS5|>fxn_values|>length),
    RESULTS5|>fxn_values.|>(x -> max(-53, x)|> log2), 
    label="N'Mono | BT "
)
plot!(
    plt2, 
    1:(RESULTS5|>lipschitz_estimates|>length),
    RESULTS5|>lipschitz_estimates.|>(x -> max(-53, x)|> log2), 
    label="N'Mono | BT"
)
plot!(
    plt3,
    1:(RESULTS5|>gradmap_values|>length),
    RESULTS5|>gradmap_values.|>(x -> max(-53, x)|> log2), 
    label="N'Mono | BT",
    dpi=400
)

plt1|>display
plt2|>display
plt3|>display