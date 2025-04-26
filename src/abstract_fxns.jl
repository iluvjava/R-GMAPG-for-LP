abstract type NsmoothFxn end

function fxn_eval(::NsmoothFxn, ::AbstractArray)::Number
    throw("Not Implemented. ")
end

function prox(::NsmoothFxn, ::Number, ::AbstractArray)::AbstractArray
    throw("Not Implemented. ")
end

function (this::NsmoothFxn)(x::AbstractArray)::Number
    return fxn_eval(this, x)
end


abstract type IndicFxn <: NsmoothFxn
end

function proj(::IndicFxn, ::AbstractArray)::AbstractArray
    throw("Not Implemented. ")
end

function prox(this::IndicFxn, ::Number, ::AbstractArray)::AbstractArray
    return proj(this, x)
end


abstract type SmoothFxn end

"""
Evaluate both gradient and function value at x. 
"""
function fxn_eval(::SmoothFxn, ::AbstractArray)::Tuple
    throw("Not implemented.")
end

"""
Get the gradient at x. 
"""
function grad(::SmoothFxn, ::AbstractArray)::AbstractArray
    throw("Not implemeneted. ")
end

"""
Get the function value at x. 
"""
function (this::SmoothFxn)(x::AbstractArray)::Number
    return fxn_eval(this, x)
end


