abstract type NsmoothFxn end


function fxn_eval(
    ::NsmoothFxn, 
    ::AbstractArray{Float64}
)::Float64
    # IMPLEMENT THIS !
    throw("Not Implemented. ")
end

function prox(
    ::NsmoothFxn, 
    ::Number, 
    ::AbstractArray{Float64}
)::AbstractArray{Float64}
    # IMPLEMENT THIS !
    throw("Not Implemented. ")
end

# Fast nonsmooth interface. 
function prox!(
    this::NsmoothFxn, 
    l::Number, 
    x::AbstractArray{Float64},
    xx::AbstractArray{Float64}
)::AbstractArray{Float64}
    copy!(xx, prox(this, l, x))
    return xx
end

function (this::NsmoothFxn)(
    x::AbstractArray{Float64}
)::Float64
    return fxn_eval(this, x)
end


# ------------------------------------------------------------------------------
abstract type IndicFxn <: NsmoothFxn
end

function proj(
    ::IndicFxn, 
    ::AbstractArray{Float64}
)::AbstractArray{Float64}
    # IMPLEMENT THIS !
    throw("Not Implemented. ")
end

function prox(
    this::IndicFxn, 
    ::Number, 
    ::AbstractArray{Float64}
)::AbstractArray{Float64}
    return proj(this, x)
end

# ------------------------------------------------------------------------------
abstract type SmoothFxn end

"""
Evaluate both gradient and function value at x. 
"""
function fxn_eval(
    this::SmoothFxn, 
    ::AbstractArray{Float64}
)::Float64
    # IMPLEMENT THIS !
    throw("Not implemente for type of `$(typeof(this))`")
end

"""
Get the gradient at x. 
"""
function grad(this::SmoothFxn, ::AbstractArray{Float64})::AbstractArray{Float64}
    # IMPLEMENT THIS !
    throw("Not implemeneted for `$(typeof(this))`. ")
end

"""
Get the function value at x. 
"""
function (this::SmoothFxn)(x::AbstractArray{Float64})::Float64
    return fxn_eval(this, x)
end

# ADAPTOR FUNCTION 
function grad!(
    this::SmoothFxn, 
    x::AbstractArray{Float64}, 
    xx::AbstractArray{Float64}
)::AbstractArray{Float64}
    copy!(xx, grad(this, x))
    return xx
end

# ==============================================================================
# FAST ABSTRACT FUNCTIONS 
# ==============================================================================

abstract type FastSmoothFxn <: SmoothFxn end

"""
Mutable the assigned reference to an abstract array to compute the 
gradient. This saves cg time. 
"""
function grad!(
    ::FastSmoothFxn, 
    ::AbstractArray{Float64}, 
    ::AbstractArray{Float64}
)::AbstractArray{Float64}
    # IMPLEMENT THIS !    
    throw("Not implemented.")
end

function (this::FastSmoothFxn)(x::AbstractArray{Float64})::Float64
    return fxn_eval(this, x)
end

# ------------------------------------------------------------------------------
abstract type FastNsmoothFxn <: NsmoothFxn end

"""
Mutable the assigned reference to an abstract array to compute the 
prox. This saves cg time. 
"""
function prox!(
    ::FastNsmoothFxn, 
    ::Number,
    ::AbstractArray{Float64}, 
    ::AbstractArray{Float64}
)::AbstractArray{Float64}
    # IMPLEMENT THIS !
    throw("Not implemented.")
end

# ------------------------------------------------------------------------------
abstract type FastIndicFxn <: FastNsmoothFxn

end

function proj!(
    ::FastIndicFxn,
    ::AbstractArray{Float64}, 
    ::AbstractArray{Float64}
)::AbstractArray{Float64}
    # IMPLEMENT THIS !
    throw("Not implemented.")
end

function prox!(
    this::FastIndicFxn,
    ::Number,
    x::AbstractArray{Float64}, 
    xx::AbstractArray{Float64}
)::AbstractArray{Float64}
    return proj!(this, x, xx)
end