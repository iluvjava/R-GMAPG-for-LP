# Euclidean Norm Squared Affine Composite ---------------------------------------
"""
f = 1/2‖Ax - b‖^2 where A is an implict linear mapping. 
Implicitly represented via: 
∇f(x) = A^TAx - A^Tb
f(x) = (1/2)<x, ∇f(x) - A^Tb> + 1/2‖d‖^2
"""
struct ImplicitAffineNormedSquared <: SmoothFxn
    lin_map::Function
    lin_map_adj::Function
    d::Number # ‖b‖^2
    c::Vector{Float64} # A^T b
    b::Vector{Float64}
    
    function ImplicitAffineNormedSquared(
        lin_map::Function, 
        lin_map_adj::Function, 
        b::Vector{Float64}
    )  
        new(lin_map, lin_map_adj, dot(b,b), lin_map_adj(b), b)
    end

end

function grad(
    this::ImplicitAffineNormedSquared, 
    x::Vector{Float64}
)::Vector{Float64}
    return this.lin_map_adj(this.lin_map(x) - this.b)
end

function fxn_eval(
    this::ImplicitAffineNormedSquared, 
    x::Vector{Float64}
)::Number
    return (1/2)*(dot(x, grad(this, x)) - dot(x, this.c) + this.d)
end

function gradient_to_fxnval(
    this::ImplicitAffineNormedSquared, 
    x::Vector{Float64}, 
    g::Vector{Float64}
)::Number
    return (1/2)*(dot(x, g) - dot(x, this.c) + this.d)
end


# FAST SQUARED LOSS ------------------------------------------------------------
"""
It's (1/2)‖Ax - b‖^2 but implementing the fast smooth function interface.
Implicitly represented via: 
∇f(x) = A^TAx - A^Tb
f(x) = (1/2)<x, ∇f(x) - A^Tb> + 1/2‖d‖^2 
"""
struct FastImplicitAffineNormedSquared <: FastSmoothFxn
    adj_comp::Function # x -> transpos(A)(A(x)) 
    adj::Function  # x -> transpose(A)(x)
    d::Float64 # ‖b‖^2
    c::Vector{Float64} # A^T b

    function FastImplicitAffineNormedSquared(
        n::Int,
        adj_comp::Function, 
        adj::Function,
        b::Vector{Float64}
    )
        new(
            adj_comp, adj, 
            dot(b, b), 
            adj(b, zeros(n))
        )
    end
end


"""
Constructor function, makes fast squared loss function with a matrix and a 
vector. 
"""
function FastImplicitAffineNormedSquared(
    M::AbstractMatrix, b::Vector{Float64}
)::FastImplicitAffineNormedSquared
    _, n = size(M)
    MTM = M'*M
    adj = let  MT = M'
        function(x, xx)
            return mul!(xx, MT, x)
        end
    end
    adj_comp = let MTM = MTM 
        function(x, xx) return mul!(xx, MTM, x) end
    end
    return FastImplicitAffineNormedSquared(n, adj_comp, adj, b)
end

function grad!(
    this::FastImplicitAffineNormedSquared, 
    x::AbstractVector{Float64}, 
    xx::AbstractVector{Float64}
)::Vector{Float64}
    ATA = this.adj_comp; ATb = this.c
    ATA(x, xx)
    xx .-= ATb
    return xx
end

function fxn_eval(
    this::FastImplicitAffineNormedSquared, 
    x::AbstractVector{Float64}
)::Float64
    return (1/2)*(dot(x, grad!(this, x, similar(x))) - dot(x, this.c) + this.d)
end

function gradient_to_fxnval(
    this::FastImplicitAffineNormedSquared, 
    x::AbstractVector{Float64}, 
    g::AbstractVector{Float64}
)::Float64
    return (1/2)*(dot(x, g) - dot(x, this.c) + this.d)
end


# The distance an affine subspace ----------------------------------------------
"""
It's the function: 
x -> min(z){‖x - z‖^2 : Az - b = 0}, where b is in rng(A). 
"""
struct DistToAffineSpaceSquared <: FastNsmoothFxn

end

"""
All quadratic function can evaluate its Bregman Divergence or function 
values at a point given its gradient and hence they are unified 
under the "Quadratic Function" type for fast implementations. They are the 
union of: 
    1. `ImplicitAffineNormedSquared`
    2. `FastImplicitAffineNormedSquared`
    3. `DistToAffineSpaceSquared`
All of them are under "SmoothFxn"
"""
GenericQuadraticFunction = Union{
    ImplicitAffineNormedSquared, 
    FastImplicitAffineNormedSquared, 
    DistToAffineSpaceSquared
}

"""
Type alias for the union of: 
    1. `FastImplicitAffineNormedSquared`, 
    2. `DistToAffineSpaceSquared`. 
All of them are under "FastSmoothFxn"
"""
FastGenericQuadraticFunction = Union{
    FastImplicitAffineNormedSquared, DistToAffineSpaceSquared
}


# ==============================================================================
# NONSMOOTH FUNCTIONS
# ==============================================================================
"""
A non-smooth function that is literally all zero, and it does nothing 
mathematically.
"""
struct ZeroFunction <: FastNsmoothFxn
    function ZeroFunction()
        return new()
    end
end

function fxn_eval(::ZeroFunction, ::AbstractArray{Float64})::Number
    return 0
end

function prox(
    ::ZeroFunction, 
    ::Number, 
    x::AbstractArray{Float64}
)::AbstractArray{Float64}
    
    return x
end

function prox!(
    ::ZeroFunction, ::Number, x::AbstractArray{Float64}, 
    xx::AbstractArray{Float64}
)::AbstractArray{Float64}
    xx .= x
    return xx 
end

# R plus indicator -------------------------------------------------------------
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
    x1 = view(x, 1:n1)
    x3 = view(x, n1+n2:n1+n2+n3) 
    y1 = max.(x1, 0)
    y3 = -max.(-x3, 0)
    return vcat(y1, x2, y3)
end


# λ|x| -------------------------------------------------------------------------
"""
x |-> λ|x|
"""
struct FastOneNorm <:FastNsmoothFxn
    lambda::Number
    function FastOneNorm(lambda::Number)
        @assert lambda > 0 "Lambda constant"*
        " for FastOneNorm has to be strictly larger than zero. "
        return new(lambda)
    end
end


function prox(
    this::FastOneNorm, 
    l::Number,
    x::AbstractArray{Float64}
)::AbstractArray{Float64}
    λ = this.lambda*l
    return @. sign(x)*max(abs(x) - λ, 0)
end


function fxn_eval(this::FastOneNorm, x::AbstractArray{Float64})::Number
    return (this.lambda*x) .|> abs |> sum
end


function prox!(
    this::FastOneNorm,
    l::Number, 
    x::AbstractArray{Float64}, 
    xx::AbstractArray{Float64}
)::AbstractArray{Float64}
    λ = this.lambda*l
    xx .= abs.(x)
    xx .-= λ
    for i in eachindex(xx)
        if xx[i] < 0
            xx[i] = 0
        end
        xx[i] = sign(x[i])*xx[i]
    end
    return xx
end

