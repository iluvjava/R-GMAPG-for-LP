
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
    r::Number
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
    ϵ = 4*eps(typeof(x0[1]))
    L̄ = L
    ρ = 2^(-1/1024)
    x⁺ = similar(x0); y⁺ = similar(x0)
    x = similar(x0); y = similar(x0)
    xg⁺ = similar(x0); yg⁺ = similar(x0)
    xg = similar(x0); yg = similar(x0)
    v = similar(x0); v⁺ = similar(x0)
    
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
    v .= x
    if results_collector|> fxn_collect
        F = gradient_to_fxnval(f, x, xg) + g(x)
    end
    α = 1; k = 0
    G = L*norm(x - x0)
    put_results!(results_collector, G, x, α ,L, fxn_val=F)
    restart_cond_met = false
    Fs = results_collector.fxn_values
    while !restart_cond_met && M >= 0
        k += 1
        M -= 1
        (L⁺, α) = ls(
            f, g, L, α, v, x, x⁺, y⁺, xg⁺, yg⁺,
            l_min=r*L̄, r=ρ
        )
        ρ = L⁺>=L ? ρ^(1/2) : ρ # upadate BT relaxation parameter. 
        L = L⁺
        L̄ = max(L, L̄); G = L*norm(x⁺ - y⁺)
        v⁺ .= x + (1/α)*(x⁺ - x)
        # Monotone enhancement here. -------------------------------------------
        if alg_settings|>monotone == 1
            F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
            if F⁺ > F + ϵ
                x⁺ .= x
                F⁺ = F
            end
        elseif alg_settings|>monotone == 2
            F⁺ = gradient_to_fxnval(f, x⁺ ,xg⁺) + g(x⁺)
            if F + ϵ < F⁺
                x⁺ .= prox(g, 1/(2L̄), x - (1/(2L̄))*xg)
                G = L̄*norm(x⁺ - x)
            else
                x .= prox(g, 1/(2L̄), x⁺ - (1/(2L̄))*xg⁺)
                G = L̄*norm(x⁺ - x)
                x⁺ .= x
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
            restart_cond_met = G < tol
        elseif alg_settings|>restart == 1
            if alg_settings|>monotone == 1
                throw(
                    "Cannot use Beck's monotone constraints"*
                    " with Gradient heuristic based restart"
                )
            end
            restart_cond_met = dot(x⁺ - y⁺, x - x⁺) > 0 && k > N
        else
            m = k - (floor(k/2)|>Int) - 1
            
            restart_cond_met = k >= N &&
            (Fs[end - m] - Fs[end])/(Fs[end - k] - Fs[end - m]) <= exp(-1) &&
            F⁺ <= Fs[end - k]
            if restart_cond_met
                println("Fx restart with k = $k")
            end
        end
        x, x⁺ = x⁺, x
        y, y⁺ = y⁺, y
        v, v⁺ = v⁺, v
        xg, xg⁺ = xg⁺, xg
        yg, yg⁺ = yg⁺, yg
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
    min_ratio=0.1,
    max_itr::Number=1000,
    tol::Number=1e-8
)::ResultsCollector 
    # MUTATING VARS
    
    M = max_itr
    N = 128 # initial minimum restart period. 
    z = x0
    j = 0
    R = Vector{Number}()
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
            M -= k
            N = max(N, k)
        else
            F, z, G, k = inner_fista_runner( 
                f, g, z, L, min_ratio, N, M, alg_settings,
                results_collector, 
                tol
            )
            M -= k
            if j == 0                
                N = max(N, k)
                j += 1
                push!(R, F)
                push!(R, results_collector.fxn_values[1])
            else 
                push!(R, F)
                if (R[end - 1] - R[end])/(R[end - 2] - R[end - 1]) > exp(-1)
                    N *= 2
                    println("Restart Strategy 2, period updated to $N. ")
                end
            end
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