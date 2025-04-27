
"""
Controls different enhancement strategies used in the FISTA. 
"""
struct AlgoSettings
    # Codes for different strategies on different enhancements. 
    "0: Armoji, 1:Chambolle Backtrack."
    line_search::Int
    "0: No monotone, 1: Beck, 2: Nesterov"
    monotone::Int
    "0: no restart, 1: Gradient heuristic, 2: Alamo's Function value restart"
    restart::Int
    function AlgoSettings(;
        line_search_strategy::Int=0, 
        monotone_strategy::Int=0, 
        restart_strategy::Int=0
    )
        @assert line_search_strategy in [0, 1] "Invalid line search"*
        " strategy code. "
        @assert monotone_strategy in [0, 1, 2] "Invalid monotone strategy code."
        @assert restart_strategy in [0, 1, 2] "Invalid monotone strategy code." 
        return new(line_search_strategy, monotone_strategy, restart_strategy)
    end
end

function line_search(this::AlgoSettings)::Int
    return this.line_search
end

function monotone(this::AlgoSettings)::Int
    return this.monotone
end

function restart(this::AlgoSettings)::Int
    return this.restart
end

"""
An objects for collecting the parameters while the algorithm is running. 
"""
mutable struct ResultsCollector
    
    fxn_collect::Bool
    
    "Function values, collection optional. "
    fxn_values::Vector{Number}
    """
    The last iterates that is going to be the solution, 
    nothing means no communication from the solver. 
    """
    last_iterate::Union{Vector{Number}, Nothing}
    "Norm of the gradient mapping, collection compulsary."
    gradmap_values::Vector{Number}
    "Alpha momentum sequence, collection compulsary. "
    alpha_sequence::Vector{Number}
    "Estimated Lipschitz constants from line search, collection compulsary. "
    lipschitz_estimates::Vector{Number}
    """
    -1: Not communication from the solver yet, 
    0: Terminated by convergence of gradient mapping,
    1: Maximum iteration exceeded, 
    """
    termination_code::Int
    
    
    function ResultsCollector(fxn_collect::Bool=true)
        this = new()
        this.fxn_collect = fxn_collect

        this.fxn_values = Vector{Number}()
        this.last_iterate = nothing
        this.gradmap_values = Vector{Number}()
        this.alpha_sequence = Vector{Number}()
        this.lipschitz_estimates = Vector{Number}()
        this.termination_code = -1
        return this
    end

end

function initial_results!(
    this::ResultsCollector, iterate::AbstractArray, fxn_val::Number=NaN
)::Nothing
    if this |> fxn_collect
        push!(this.fxn_values, fxn_val)
    end
    this.last_iterate = iterate
    return nothing
end

function put_results!(
    this::ResultsCollector, 
    gradmap_values::Number, 
    last_iterate::AbstractArray,
    alpha::Number, 
    lipschitz_estimates::Number; 
    fxn_val::Number=NaN
)::Nothing
    if this |> fxn_collect
        push!(this.fxn_values, fxn_val)
    end
    push!(this.gradmap_values, gradmap_values)
    this.last_iterate = last_iterate
    push!(this.alpha_sequence, alpha)
    push!(this.lipschitz_estimates, lipschitz_estimates)
    return nothing
end

function exit_flag!(this::ResultsCollector, flag::Int)
    this.termination_code = flag
    return nothing
end

function fxn_collect(this::ResultsCollector)::Bool
    return this.fxn_collect
end

function fxn_values(this::ResultsCollector)::AbstractVector
    return this.fxn_values
end

function gradmap_values(this::ResultsCollector)::AbstractVector
    return this.gradmap_values
end

function lipschitz_estimates(this::ResultsCollector)::AbstractVector
    return this.lipschitz_estimates
end

function alpha_sequence(this::ResultsCollector)::AbstractVector
    return this.alpha_sequence
end