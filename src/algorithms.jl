
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
    r::Number=2^(-1/1024)
)::NTuple
    throw("Not implemented yet. ")
end

"""
Specialize Armijo line search routine for quadratic function. 
"""
function armijo_ls(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray,
    x::AbstractArray;
    kwargs...
)::NTuple{6, Any}
    α = alpha  
    α = (1/2)*(α*sqrt(α^2 + 4) - α^2)
    y = α*v + (1 - α)*x
    y′ = grad(f, y)
    p′ = similar(y); p = similar(y)
    for _ in 1:53
        p = prox(g, 1/L, y - (1/L)*y′)
        p′ = grad(f, p)
        b = dot(y′ - p′, y - p)
        if b <= L*dot(p - y, p - y)
            break
        end
        L = 2*L
    end
    return L, α, p, y, p′, y′
end


"""
Chambolle's back tracking LS without any strong convexity modifiers. 
Specialize for normed quadratic functions. 
"""
function backtrack_ls(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray,
    x::AbstractArray;
    l_min::Number,
    r::Number=2^(-1/512)
)::NTuple{6, Any}
    L⁺ = max(L*r, l_min)
    α = alpha
    y = similar(x); y′ = similar(x)
    p = similar(x); p′ = similar(x)
    for i in 0:53
        α = (1/2)*(α*sqrt(α^2 + 4(L/L⁺)) - α^2)
        y = α*v + (1 - α)*x
        y′ = grad(f, y)
        p = prox(g, 1/L⁺, y - (1/L⁺)*y′)
        p′ = grad(f, p)
        b = dot(y′ - p′, y - p)
        L⁺ = L⁺*2^i
        if b <= L⁺*dot(p - y, p - y)
            break
        end
    end
    return L⁺, α, p, y, p′, y′
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
    min_ratio=0.4,
    max_itr::Number=1000,
    tol::Number=1e-8
)::ResultsCollector
    if results_collector|> fxn_collect
        F = f(x0) + g(x0)
        initial_results!(results_collector, x0, F)
    else
        F = NaN
        initial_results!(results_collector, x0)
    end
    (L, _, x, _, xg, yg) = armijo_ls(f, g, L, 0, x0, x0) 
    v = x
    
    if results_collector|> fxn_collect
        F = gradient_to_fxnval(f, xg) + g(x)
    end
    initial_results!(results_collector, x, F)

    ls = alg_settings|>line_search == 0 ? armijo_ls : backtrack_ls
    exitFlag = 0
    r = min_ratio
    L̄ = L
    α = 1
    
    for k in 1:max_itr
        (L, α, x⁺, y⁺, xg⁺, yg⁺) = ls(f, g, L, α, v, x, l_min=r*L̄)
        L̄ = max(L, L̄)
        G = L̄*norm(x⁺ - y⁺)
        # Monotone enhancement here. -------------------------------------------
        if alg_settings|>monotone == 1
            
            F⁺ = gradient_to_fxnval(f, xg⁺) + g(x⁺)
            if F⁺ > F
                x⁺ = x
            end
            v⁺ = x + (1/α)*(x⁺ - x)
        elseif alg_settings|>monotone == 2
            
            F⁺ = gradient_to_fxnval(f, xg⁺) + g(x⁺)
            if F⁺ > F
                x⁺ = x; xg⁺ = xg; F⁺ = F
            end
            x⁺ = prox(g, 1/L̄, x⁺ - (1/L̄)*xg⁺)
            v⁺ = x + (1/α)*(x⁺ - x)
            
        else
            if results_collector|> fxn_collect 
                F⁺ = gradient_to_fxnval(f, xg⁺) + g(x⁺)
            end
            v⁺ = x + (1/α)*(x⁺ - x)
        
        end
        # Recording results here. ----------------------------------------------
        put_results!(results_collector, G, x⁺, α, L, fxn_val=F⁺)
        # Restart --------------------------------------------------------------

        # check terminations. --------------------------------------------------
        if G < tol
            break
        end
        v, x = v⁺, x⁺
        xg, yg = xg⁺, yg⁺
        F = F⁺
        if k == max_itr
            exitFlag = 1
        end
    end
    exit_flag!(results_collector, exitFlag)
    return results_collector
end