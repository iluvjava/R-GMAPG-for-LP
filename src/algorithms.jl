
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
    f::GenericQuadraticFunction, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray,
    x::AbstractArray,
    x_plus::AbstractArray, 
    y_plus::AbstractArray,
    xg_plus::AbstractArray,
    yg_plus::AbstractArray,
    z::AbstractArray; # for signaure consistency with fast implementations. 
    kwargs...
)::Tuple
    α = alpha  
    α = (1/2)*(α*sqrt(α^2 + 4) - α^2)
    # y_plus .= α*v + (1 - α)*x
        copy!(z, v)
        z .*= α
        copy!(y_plus, x)
        y_plus .*= 1 - α
        y_plus .+= z
    yg_plus .= grad(f, y_plus)
    for _ in 1:53
        # x_plus .= prox(g, 1/L, y_plus - (1/L)*yg_plus) # Fast alternative:
        copy!(z, yg_plus)
        z .*= -1/L
        z .+= y_plus
        copy!(x_plus, prox(g, 1/L, z))
        # xg_plus .= grad(f, x_plus) # Faster alternative
        copy!(xg_plus, grad(f, x_plus))
        # b = dot(yg_plus - xg_plus, y_plus - x_plus); # Fast alternative: 
        b = dot(yg_plus, y_plus) - dot(yg_plus, x_plus)
        b += -dot(xg_plus, y_plus) + dot(xg_plus, x_plus)
        copy!(z, x_plus)
        z .-= y_plus
        if b <= L*dot(z, z)
            break
        end
        L = 2*L
    end
    return L, α
end



"""
Specialize armijo line search routine for fast quadratic and nonsmooth
functions. 
"""
function armijo_ls!(
    f::FastGenericQuadraticFunction, # !! For fast function. 
    g::FastNsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::Vector{Float64},
    x::Vector{Float64},
    x_plus::Vector{Float64}, 
    y_plus::Vector{Float64},
    xg_plus::Vector{Float64},
    yg_plus::Vector{Float64}, 
    z::Vector{Float64};
    kwargs...
)::Tuple
    α = alpha  
    α = (α*sqrt(α^2 + 4) - α^2)/2
    
    copy!(y_plus, v)  # y_plus .= v
    y_plus .*= α
    copy!(z, x); z .*= (1 - α)
    y_plus .+= z
    grad!(f, y_plus, yg_plus)
    for _ in 1:53
        copy!(z, yg_plus) # temp_vec1 .= yg_plus
        z .*= -1/L
        z .+= y_plus
        prox!(g, 1/L, z, x_plus)
        grad!(f, x_plus, xg_plus)
        b = dot(yg_plus, y_plus) - dot(yg_plus, x_plus) - 
            dot(xg_plus, y_plus) + dot(xg_plus, x_plus)
        copy!(z, x_plus) 
        z .-= y_plus
        if b <= L*dot(z, z)
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
    f::GenericQuadraticFunction, 
    g::NsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray{Float64},
    x::AbstractArray{Float64},
    x_plus::AbstractArray{Float64}, 
    y_plus::AbstractArray{Float64},
    xg_plus::AbstractArray{Float64},
    yg_plus::AbstractArray{Float64},
    z::AbstractArray{Float64};  # for signaure consistency with fast implementations. 
    l_min::Number,
    r::Number
)::Tuple
    L⁺ = max(L*r, l_min)
    α = alpha
    y = y_plus; y′ = yg_plus
    p = x_plus; p′ = xg_plus
    for i in 0:53
        α = (α*sqrt(α^2 + 4(L/L⁺)) - α^2)/2
        # copy!(y, α*v + (1 - α)*x)
        
        copy!(z, v)
        z .*= α
        copy!(y, x)
        y .*= (1 - α)
        y .+= z

        copy!(y′, grad(f, y))
        copy!(z, y′)
        z .*= -1/L⁺
        z .+= y 
        copy!(p, prox(g, 1/L⁺, z))
        copy!(p′, grad(f, p))
        b = dot(y′, y) - dot(y′, p) - dot(p′, y) + dot(p′, p)
        L⁺ = L⁺*2^i
        
        copy!(z, p)
        z .-= y;
        if b <= L⁺*dot(z, z)
            break
        end
    end
    return L⁺, α
end


function backtrack_ls!(
    f::FastGenericQuadraticFunction, 
    g::FastNsmoothFxn, 
    L::Number, 
    alpha::Number,
    v::AbstractArray{Float64},
    x::AbstractArray{Float64},
    x_plus::AbstractArray{Float64}, 
    y_plus::AbstractArray{Float64},
    xg_plus::AbstractArray{Float64},
    yg_plus::AbstractArray{Float64},
    z::AbstractArray{Float64};  # for signaure consistency with fast implementations. 
    l_min::Number,
    r::Number
)::Tuple
    L⁺ = max(L*r, l_min)
    α = alpha
    y = y_plus; y′ = yg_plus
    p = x_plus; p′ = xg_plus

    for i in 0:53
        α = (α*sqrt(α^2 + 4(L/L⁺)) - α^2)/2
        # copy!(y, α*v + (1 - α)*x) # faster alternative: 
        copy!(z, v)
        z .*= α
        copy!(y, x)
        y .*= (1 - α)
        y .+= z
        grad!(f, y, y′) # y′ <- ∇f(y)
        copy!(z, y′)
        z .*= -1/L⁺
        z .+= y 
        prox!(g, 1/L⁺, z, p) # p <- prox(g, p - ∇f(p)/L)
        grad!(f, p, p′)  # p′ <- ∇f(p)
        # b = dot(y′ - p′, y - p)
        b = dot(y′, y) - dot(y′, p) - dot(p′, y) + dot(p′, p)
        L⁺ = L⁺*2^i
        copy!(z, p)
        z .-= y;
        if b <= L⁺*dot(z, z) # L⁺‖p - y‖^2
            break
        end
    end
    return L⁺, α
end



"""
A specialized Inner fista runner for normed squared quadratic.
"""
function inner_fista_runner(
    f::GenericQuadraticFunction, 
    g::NsmoothFxn,
    x0::AbstractArray{Float64}, 
    L::Number, # Initial Lipschitz constant guess.  
    r::Number, # Relaxation parameters for backtracking line search. 
    N::Int, # Minimum iteration needed. 
    M::Int, # Maximum iteration allowed by outter loop. 
    alg_settings::AlgoSettings, 
    results_collector::ResultsCollector, 
    tol::Number
):: NTuple{4, Any}
    ls = alg_settings|>line_search == 0 ? armijo_ls! : backtrack_ls!
    ϵ = eps(typeof(x0[1]))
    L̄ = L
    ρ = 2^(-1/1024)
    x⁺ = similar(x0); y⁺ = similar(x0)
    x = similar(x0); y = similar(x0)
    xg⁺ = similar(x0); yg⁺ = similar(x0)
    xg = similar(x0); yg = similar(x0)
    v = similar(x0); v⁺ = similar(x0)
    z1 = similar(x0) # temporary storage
    z2 = similar(x0) # temporary storage. 
    
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
    (L, _) = armijo_ls!(f, g, L, 0, x0, x0, x, y, xg, yg, z1)
    copyto!(v, x)
    if results_collector|> fxn_collect
        F = gradient_to_fxnval(f, x, xg) + g(x)
    end
    α = 1; k = 0
    G = L*norm(x - x0)
    put_results!(results_collector, G, α ,L, fxn_val=F)
    restart_cond_met = false
    Fs = results_collector.fxn_values
    while !restart_cond_met && M >= 0
        k += 1
        M -= 1
        (L⁺, α) = ls(
            f, g, L, α, v, x, x⁺, y⁺, xg⁺, yg⁺, z1,
            l_min=r*L̄, r=ρ
        )
        ρ = L⁺>=L ? ρ^(1/2) : ρ # upadate BT relaxation parameter. 
        L = L⁺
        copy!(z2, x⁺)
        z2 .-= y⁺ 
        L̄ = max(L, L̄); G = L*norm(z2)
        copy!(v⁺, x⁺); v⁺ .-= x; v⁺ .*= 1/α; v⁺ .+= x # v⁺ = x + (1/α)*(x⁺ - x) 
        # Monotone enhancement here. -------------------------------------------
        if alg_settings|>monotone == 1
            F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
            if F⁺ > F + ϵ
                copy!(x⁺, x) # x⁺ .= x
                F⁺ = F
            end
        elseif alg_settings|>monotone == 2
            F⁺ = gradient_to_fxnval(f, x⁺ ,xg⁺) + g(x⁺)
            if F + ϵ < F⁺
                # x⁺ .= prox(g, 1/(2L̄), x - (1/(2L̄))*xg)
                # Speedy implementations: 
                copy!(z2, xg)
                z2 .*= -1/L̄
                z2 .+= x
                copy!(x⁺, prox(g, 1/L̄, z2))
                G = L̄*norm(x⁺ - x)
            else
                # x .= prox(g, 1/(2L̄), x⁺ - (1/(2L̄))*xg⁺)
                # Speedy implementations: 
                copy!(z2, xg⁺)
                z2 .*= -1/L̄
                z2 .+= x⁺
                copy!(x, prox(g, 1/L̄, z2))
                G = L̄*norm(x⁺ - x)
                copy!(x⁺, x)

            end
            copy!(xg⁺, grad(f, x⁺)) # speed alternative for xg⁺ .= grad(f, x⁺)
            F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
        else
            if results_collector|> fxn_collect || 
            alg_settings|>restart >= 2
                F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
            end
        end
        # Recording results here. ----------------------------------------------
        put_results!(results_collector, G, α, L, fxn_val=F⁺)
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

            # Exit anyway and let the outer loop break. 
            restart_cond_met |= G < tol
        end
        x, x⁺ = x⁺, x
        y, y⁺ = y⁺, y
        v, v⁺ = v⁺, v
        xg, xg⁺ = xg⁺, xg
        yg, yg⁺ = yg⁺, yg
        F = F⁺
    end
    # store the result! 
    last_iterate!(results_collector, x)
    return F, x, G, k
end




"""
FISTA, specialized for squared norm composite of linear mapping. 
"""
function fista(
    f::GenericQuadraticFunction, 
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
        # for all restarts, break if tolerance reached or maximum total 
        # iteration reached. 
        if G < tol || M <= 0   
            break
        end
    end
    return results_collector
end




"""
A dead simple FISTA for Sanity check. 
"""
function fista_sanity_check(
    f::GenericQuadraticFunction, 
    g::NsmoothFxn, 
    x0::Vector{Float64};
    L::Number=1,
    alg_settings::AlgoSettings=AlgoSettings(), 
    results_collector::ResultsCollector=ResultsCollector(),
    max_itr::Number=1000,
    r::Number =0.3,
    tol::Number=1e-8
)
    ls = alg_settings|>line_search == 0 ? armijo_ls! : backtrack_ls!
    M = max_itr
    x⁺ = similar(x0); y⁺ = similar(x0)
    x = similar(x0); y = similar(x0)
    xg⁺ = similar(x0); yg⁺ = similar(x0)
    xg = similar(x0); yg = similar(x0)
    v = similar(x0); v⁺ = similar(x0)
    z1 = similar(x0)
    z2 = similar(x0)
    ρ = 2^(-1/1024)

    # First iterates is just a proximal gradient step --------------------------
    if results_collector|> fxn_collect
        F = f(x0) + g(x0)
    end
    initial_results!(results_collector, x0, F)
    (L, _) = armijo_ls!(f, g, L, 0, x0, x0, x, y, xg, yg, z1)
    copy!(v, x)
    if results_collector|> fxn_collect
        F = gradient_to_fxnval(f, x, xg) + g(x)
    end
    α = 1
    G = norm(L*(x - x0))
    put_results!(results_collector, G, α, L, fxn_val=F)
    while M >= 0 
        M -= 1
        (L⁺, α) = ls(
            f, g, L, α, v, x, x⁺, y⁺, xg⁺, yg⁺, z1,
            l_min=r*L, r=ρ
        )
        ρ = L⁺>=L ? ρ^(1/2) : ρ # upadate BT relaxation parameter. 
        L = L⁺
        copy!(z2, x⁺); z2 .-= x
        G = L*norm(z2)
        # Fast implementation of  v⁺ =  x + (1/α)*(x⁺ - x) below: 
        copy!(v⁺, x⁺) 
        v⁺ .-= x
        v⁺ .*= 1/α
        v⁺ .+= x
        F⁺ = gradient_to_fxnval(f, x⁺, xg⁺) + g(x⁺)
        put_results!(results_collector, G, α, L, fxn_val=F⁺)
        if G < tol
            break
        end
        x, x⁺ = x⁺, x  # ref swapping. 
        y, y⁺ = y⁺, y
        v, v⁺ = v⁺, v
        F = F⁺
    end
    last_iterate!(results_collector, x)
    
    return results_collector
end