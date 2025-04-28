
function armijo_ls(f::SmoothFxn, g::NsmoothFxn, y::AbstractArray)
    throw("not implemented yet. ")
end

function backtrack_ls(
    f::SmoothFxn, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray,
    x::AbstractArray;
    l_min::Number,
    r::Number=1
)::NTuple
    throw("Not implemented yet. ")
end

"""
Specialize Armijo line search routine for quadratic function. 
"""
function armijo_ls!(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray,
    x::AbstractArray,
    x_plus::AbstractArray, 
    y_plus::AbstractArray,
    xg_plus::AbstractArray,
    yg_plus::AbstractArray;
    kwargs...
)::Tuple
    α = alpha  
    α = (1/2)*(α*sqrt(α^2 + 4) - α^2)
    y_plus .= α*v + (1 - α)*x
    yg_plus .= grad(f, y_plus)
    for _ in 1:53
        x_plus .= prox(g, 1/L, y_plus - (1/L)*yg_plus)
        xg_plus .= grad(f, x_plus)
        b = dot(yg_plus - xg_plus, y_plus - x_plus)
        if b <= L*dot(x_plus - y_plus, x_plus - y_plus)
            break
        end
        L = 2*L
    end
    return L, α
end


"""
Chambolle's back tracking LS without any strong convexity modifiers. 
Specialize for normed quadratic functions. 
"""
function backtrack_ls!(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray,
    x::AbstractArray,
    x_plus::AbstractArray, 
    y_plus::AbstractArray,
    xg_plus::AbstractArray,
    yg_plus::AbstractArray;
    l_min::Number,
    r::Number=2^(-1/1024)
)::Tuple
    L⁺ = max(L*r, l_min)
    α = alpha
    y = y_plus; y′ = yg_plus
    p = x_plus; p′ = xg_plus
    for i in 0:53
        α = (1/2)*(α*sqrt(α^2 + 4(L/L⁺)) - α^2)
        y .= α*v + (1 - α)*x
        y′ .= grad(f, y)
        p .= prox(g, 1/L⁺, y - (1/L⁺)*y′)
        p′ .= grad(f, p)
        b = dot(y′ - p′, y - p)
        L⁺ = L⁺*2^i
        if b <= L⁺*dot(p - y, p - y)
            break
        end
    end
    return L⁺, α
end


"""
A specialized Inner fista runner for normed squared quadratic.
"""
function inner_fista_runner(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn,
    x0::AbstractArray, 
    L::Number, # Initial Lipschitz constant guess.  
    r::Number, # Relaxation parameters for backtracking line search. 
    N::Int, # Minimum iteration needed. 
    M::Int, # Maximum iteration allowed by outter loop. 
    alg_settings::AlgoSettings, 
    results_collector::ResultsCollector, 
    tol::Number
):: NTuple{4, Any}
    ls = alg_settings|>line_search == 0 ? armijo_ls! : backtrack_ls!
    L̄ = L
    x⁺ = similar(x0); y⁺ = similar(x0)
    x = similar(x0); y = similar(x0)
    xg⁺ = similar(x0); yg⁺ = similar(x0)
    xg = similar(x0); yg = similar(x0)
    v = similar(x0)
    
    # First iterates is just a proximal gradient step --------------------------
    if results_collector|> fxn_collect || 
       alg_settings|>monotone != 0 ||
       alg_settings|>restart >= 2 
        F = f(x0) + g(x0)
        initial_results!(results_collector, x0, F)
    else
        F = NaN
        initial_results!(results_collector, x0)
    end
    (L, _) = armijo_ls!(f, g, L, 0, x0, x0, x, y, xg, yg)
    v = x
    if results_collector|> fxn_collect
        F = gradient_to_fxnval(f, x, xg) + g(x)
    end
    initial_results!(results_collector, x, F)
    
    # restart related parameters -------------------------------------------
    k = 0; restart_cond_met = false
    F0 = F; G = L*(x - x0)
    α = 1
    while !restart_cond_met && M >= 0
        k += 1
        M -= 1
        (L, α) = ls(
            f, g, L, α, v, x, x⁺, y⁺, xg⁺, yg⁺,
            l_min=r*L̄
        )
        L̄ = max(2*L, L̄); G = L*norm(x⁺ - y⁺)
        v .= x + (1/α)*(x⁺ - x)
        # Monotone enhancement here. ---------------------------------------
        if alg_settings|>monotone == 1
            F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
            if F⁺ > F
                x⁺ = x|>copy
                F⁺ = F
            end
        elseif alg_settings|>monotone == 2
            F⁺ = gradient_to_fxnval(f, x⁺ ,xg⁺) + g(x⁺)
            if F < F⁺
                x⁺ .= prox(g, 1/L̄, x - (1/L̄)*xg)
                G = L̄*norm(x⁺ - x)
            else
                x̂ = prox(g, 1/L̄, x⁺ - (1/L̄)*xg⁺)
                G = L̄*norm(x⁺ - x̂)
                x⁺ = x̂
            end
            xg⁺ .= grad(f, x⁺)
            F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
        else
            if results_collector|> fxn_collect || 
            alg_settings|>restart >= 2
                F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
            end
        end
        # Recording results here. ----------------------------------------------
        put_results!(results_collector, G, x⁺, α, L, fxn_val=F⁺)
        
        # check restart conditions here ----------------------------------------
        if alg_settings|>restart == 0
            if G < tol
                restart_cond_met = true
            end
        elseif alg_settings|>restart == 1
            if alg_settings|>monotone == 1
                throw(
                    "Cannot use Beck's monotone constraints"*
                    " with Gradient heuristic based restart"
                )
            end
            if dot(x⁺ - y⁺, x - x⁺) > 0 && k > N
                restart_cond_met = true
            end
        else
            throw("Not implemeneted")
        end
        yg = yg⁺|>copy; xg = xg⁺|>copy
        x = x⁺|>copy; y = y⁺|>copy
        F = F⁺
    end

    return F, x, G, k
end

"""
FISTA, specialized for squared norm composite of linear mapping. 
"""
function fista(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    x0::AbstractArray;
    L::Number=1,
    alg_settings::AlgoSettings=AlgoSettings(), 
    results_collector::ResultsCollector=ResultsCollector(),
    min_ratio=0.2,
    max_itr::Number=1000,
    tol::Number=1e-8
)::ResultsCollector 
    # MUTATING VARS
    
    M = max_itr
    N = 10 # minimum restart period. 
    z = x0
    while M >= 0
        if alg_settings|>restart == 0
            F, z, G, k = inner_fista_runner( 
                f, g, z, L, min_ratio, M, M, alg_settings,
                results_collector, 
                tol
            )
            # No restart, run once then it runs to finish 
            break
        elseif alg_settings|>restart == 1
            # Gradient Heuristic restart. 
            F, z, G, k = inner_fista_runner( 
                f, g, z, L, min_ratio, N, M, alg_settings,
                results_collector, 
                tol
            )
            println("rs period: $k")
            M -= k
            N = max(N, k)
        else
            throw("Not implemeneted")
        end

        if G < tol           
            break
        end
    end
    
    return results_collector
end


"""
A dead simple FISTA for Sanity check. 
"""
function fista_sanity_check(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    x0::AbstractArray;
    L::Number=1,
    alg_settings::AlgoSettings=AlgoSettings(), 
    results_collector::ResultsCollector=ResultsCollector(),
    max_itr::Number=1000,
    tol::Number=1e-8
)
    M = max_itr
    x⁺ = similar(x0); y⁺ = similar(x0)
    x = similar(x0); y = similar(x0)
    xg⁺ = similar(x0); yg⁺ = similar(x0)
    xg = similar(x0); yg = similar(x0)
    v = similar(x0); v⁺ = similar(x0)

    # First iterates is just a proximal gradient step --------------------------
    if results_collector|> fxn_collect
        F = f(x0) + g(x0)
    end
    initial_results!(results_collector, x0, F)
    (L, _) = armijo_ls!(f, g, L, 0, x0, x0, x, y, xg, yg)
    v = x
    if results_collector|> fxn_collect
        F = gradient_to_fxnval(f, x, xg) + g(x)
    end
    α = 1
    G = norm(L*(x - x0))
    put_results!(results_collector, G, x, α, L, fxn_val=F)
    while M >= 0 
        M -= 1
        (L, α) = armijo_ls!(f, g, L, α, v, x, x⁺, y⁺, xg⁺, yg⁺)
        G = L*norm(x⁺ - y⁺)
        v⁺ = x + (1/α)*(x⁺ - x)
        F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
        put_results!(results_collector, G, x⁺, α, L, fxn_val=F⁺)
        if G < tol
            break
        end
        x, x⁺ = x⁺, x
        y, y⁺ = y⁺, y
        v, v⁺ = v⁺, v
        F = F⁺
    end
    
    return results_collector
end