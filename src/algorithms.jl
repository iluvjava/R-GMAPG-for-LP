
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
    r::Number=2^(-1/1024)
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
        b = (1/2)*dot(y′ - p′, y - p)
        L⁺ = L⁺*2^i
        if b <= (L⁺/2)*dot(p - y, p - y)
            break
        end
    end
    return L, α, p, y, p′, y′
end


"""
FISTA, specialized for squared norm composite of linear mapping. 
"""
function fista(
    f::ENormSquaredViaLinMapImplicit, 
    g::NsmoothFxn, 
    x0::AbstractArray;
    L::Number=1,
    mono_strategies=0,
    ls::Function=armijo_ls,
    min_ratio=0.1,
    max_itr::Number=1000,
    tol::Number=1e-10
)
    (L, _, x, y, xg, yg) = armijo_ls(f, g, L, 0, x0, x0)  # A simple prox gradient step 
    v = x0
    exitFlag = 0
    r = min_ratio
    L̄ = L
    α = 1
    for k in 1:max_itr
        (L, α, x⁺, y⁺, xg⁺, yg⁺) = ls(f, g, L, α, v, x, l_min=r*L̄)
        v⁺ = x + (1/α)*(x⁺ - x)
        println(x⁺)
        L̄ = max(L, L̄)
        # Monotone constraints here. 
        G = L̄*norm(x⁺ - y⁺)
        if G < tol
            break
        end
        v, x, y = v⁺, x⁺, y⁺
        xg, yg = xg⁺, yg⁺
        if k == max_itr
            exitFlag = 1
        end
    end
    return x
end