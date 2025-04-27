
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
)::NTuple{6, Any}
    α = alpha  
    α = (1/2)*(α*sqrt(α^2 + 4) - α^2)
    y_plus = α*v + (1 - α)*x
    yg_plus = grad(f, y_plus)
    for _ in 1:53
        x_plus = prox(g, 1/L, y_plus - (1/L)*yg_plus)
        xg_plus = grad(f, x_plus)
        b = dot(yg_plus - xg_plus, y_plus - x_plus)
        if b <= L*dot(x_plus - y_plus, x_plus - y_plus)
            break
        end
        L = 2*L
    end
    return L, α, x_plus, y_plus, xg_plus, yg_plus
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
)::NTuple{6, Any}
    L⁺ = max(L*r, l_min)
    α = alpha
    # y = view(y_plus, :); y′ = view(yg_plus, :)
    # p = view(x_plus, :); p′ = view(xg_plus, :)
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
    min_ratio=0.2,
    max_itr::Number=1000,
    tol::Number=1e-8
)::ResultsCollector
    ls = alg_settings|>line_search == 0 ? armijo_ls! : backtrack_ls!
    r = min_ratio; L̄ = L; α = 1
    M = max_itr
    # for speeds 
    x⁺ = similar(x0); y⁺ = similar(x0)
    x = similar(x0); y = similar(x0)
    xg⁺ = similar(x0); yg⁺ = similar(x0)
    xg = similar(x0); yg = similar(x0)
    function inner_fista_runner(N::Int, x0)
        if results_collector|> fxn_collect || 
        alg_settings|>monotone != 0 ||
        alg_settings|>restart >= 2
            F = f(x0) + g(x0)
            initial_results!(results_collector, x0, F)
        else
            F = NaN
            initial_results!(results_collector, x0)
        end
        (L, _, x, _, xg, yg) = armijo_ls!(f, g, L, 0, x0, x0, x, y, xg, yg) 
        v = x
        
        if results_collector|> fxn_collect
            F = gradient_to_fxnval(f, x, xg) + g(x)
        end
        initial_results!(results_collector, x, F)
        # restart related parameters -------------------------------------------
        k = 0; restart_cond_met = false
        F0 = F; G = L*(x - x0)
        while (k <= N && !restart_cond_met) || M >= 0
            k += 1; M -= 1
            (L, α, x⁺, y⁺, xg⁺, yg⁺) = ls(
                f, g, L, α, v, x, x⁺, y⁺, xg⁺, yg⁺,
                l_min=r*L̄
            )
            L̄ = max(2*L, L̄); G = L*norm(x⁺ - y⁺)
            v .= x + (1/α)*(x⁺ - x)
            # Monotone enhancement here. ---------------------------------------
            if alg_settings|>monotone == 1
                F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
                if F⁺ > F
                    x⁺, x = x, x⁺
                    F⁺ = F
                end
            elseif alg_settings|>monotone == 2
                F⁺ = gradient_to_fxnval(f,x⁺ ,xg⁺) + g(x⁺)
                if F < F⁺
                    x⁺ = prox(g, 1/L̄, x - (1/L̄)*xg)
                    G = L̄*norm(x⁺ - x)
                else
                    x = prox(g, 1/L̄, x⁺ - (1/L̄)*xg⁺)
                    G = L̄*norm(x - x⁺)
                    x⁺ = x
                end
                xg⁺ = grad(f, x⁺)
                F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
            else
                if results_collector|> fxn_collect || 
                alg_settings|>restart >= 2
                    F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
                end
            end
            # Recording results here. ------------------------------------------
            put_results!(results_collector, G, x⁺, α, L, fxn_val=F⁺)
            # check restart conditions here ------------------------------------
            if alg_settings|>restart == 0
                if G < tol
                    break
                end
            elseif alg_settings|>restart == 1
                if alg_settings|>monotone == 1
                    throw(
                        "Cannot use Beck's monotone constraints"*
                        " with Gradient heuristic based restart"
                    )
                end
                if dot(x - y⁺, x⁺ - x) >= 0
                    restart_cond_met = true
                end
            else
                throw("Not implemeneted")
            end
            yg = yg⁺; xg = xg⁺
            x = x⁺
            F = F⁺
        end
        return F, x, G
    end
    n = 0 # restart period. 
    while M >= 0
        F, x = inner_fista_runner(n, x0)
        if alg_settings|>restart == 0
            break
        elseif alg_settings|>restart == 1
            F, x, G = inner_fista_runner(n, x)
            if G < tol
                break
            end
        else
            throw("Not implemeneted")
        end
        break
    end
    return results_collector
end


function fista_restart()
end