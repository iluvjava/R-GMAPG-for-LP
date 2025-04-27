"""
f = 1/2‖Ax - b‖^2 where A is an implict linear mapping. 
Implicitly represented via: 
∇f(x) = A^TAx - A^Tb
f(x) = (1/2)<x, ∇f(x) - A^Tb> + 1/2‖d‖^2
"""

struct ENormSquaredViaLinMapImplicit <: SmoothFxn
    linMap::Function
    linMapAdjoin::Function
    d::Number
    c::AbstractVector
    
    function ENormSquaredViaLinMapImplicit(
        lin_map::Function, 
        lin_map_adj::Function, 
        b::AbstractVector
    )  
        new(lin_map, lin_map_adj, dot(b,b), lin_map_adj(b))
    end

end

function grad(this::ENormSquaredViaLinMapImplicit, x::AbstractVector)::AbstractVector
    return this.linMapAdjoin(this.linMap(x)) - this.c
end

function fxn_eval(this::ENormSquaredViaLinMapImplicit, x::AbstractVector)::Number
    return (1/2)*(dot(x, grad(this, x) - this.c) + this.d)
end

function gradient_to_fxnval(
    this::ENormSquaredViaLinMapImplicit, 
    x::AbstractVector, 
    g::AbstractVector
)::Number
    return (1/2)*(dot(x, g - this.c) + this.d)
end


# ==============================================================================
# NONSMOOTH FUNCTIONS
# ==============================================================================

"""
A non-smooth function that is literally all zero, and it does nothing 
mathematically.
"""
struct ZeroFunction <: NsmoothFxn
    function ZeroFunction()
        return new()
    end
end

function fxn_eval(::ZeroFunction, ::AbstractArray)::Number
    return 0
end

function prox(::ZeroFunction, ::Number, x::AbstractArray)
    return x
end


"""
Indicator function of the Positive Cone. 
"""
struct IndicPositiveCone <: NsmoothFxn
    function IndicPositiveCone()
        return new()
    end
end


function prox(::IndicPositiveCone, x::AbstractArray)::AbstractArray
    return max.(x, 0)
end



"""
Indicator of (positive cone) × (whole space) × (negative cone)
"""
struct IndicConeCrossed <: NsmoothFxn
    n1::Int
    n2::Int
    n3::Int
    function IndicConeCrossed(n1::Int, n2::Int, n3::Int)
        @assert n1 > 0 "n1 = $n, it should > 0. "
        @assert n2 > 0 "n2 = $n, it should > 0. "
        @assert n3 > 0 "n3 = $n, it should > 0. "
        return new(n1, n2, n3)
    end
end

function proj(this::IndicConeCrossed, x::AbstractArray)::AbstractArray
    n1 = this.n1
    n3 = this.n3
    x1 = view(x, 1:n1-1)
    x3 = view(x, n2:n3-1) 
    y1 = max.(x1, 0)
    y3 = -max.(-x3, 0)
    return vcat(y1, x2, y3)
end

"""
x |-> λ|x|
"""
struct OneNorm <:NsmoothFxn
    lambda::Number
    function NsmoothFxn(lambda::Number)
        return new(lambda)
    end
end


function fxn_eval(this::OneNorm, x::AbstractArray)::Number
    λ = this.lambda
    return abs(λ*x)
end

function prox(this::OneNorm, l::Number,x::AbstractArray)::AbstractArray
    λ = this.lambda*l
    return @. sign(x)*max(abs(x) - t*λ, 0)    
end

