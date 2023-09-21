"""
Datatypes to maintain running statistics (mean, min/max, variance, etc.) of data
without storing the entire collection. Useful for streamed data. 
"""
module RunningStatistics

using LinearAlgebra: dot

import Base: push!, eltype, length, isempty, merge
import Statistics: mean, var, std		# maximum, minimum

export RunningMeanVar
export mean, var, std, uncert  			# maximum, minimum


# abstract type AbstractRunningStatistic{T,R,S} end


"""
	RunningMeanVar{T}()

An object that keeps a running mean and variance of sequentially input data
of type `T`.  It is used like a collection that implements [`length`](@ref), [`push!`](@ref),
[`mean`](@ref), [`var`](@ref), [`std`](@ref), and [`uncert`](@ref). 

The type `T` must support addition/subtraction, scaling by a Float64, and
two-argument `dot`.
"""
mutable struct RunningMeanVar{T,R,S}
	count::Int
	mean::Union{R, Nothing}
	sumsq::Union{S, Nothing}
	# min::Union{T, Nothing}
	# max::Union{T, Nothing}

	function RunningMeanVar{T}() where {T}
		R = Base.return_types(*, (T, Float64))[1]
		S = Base.return_types(realdot, (R, R))[1]		# or promote_type(eltype(T), Float64)
		new{T,R,S}(0, nothing, nothing)
	end
end

# internal function only used to help determine type
realdot(a,b) = real(dot(a,b))


# Basic properties
eltype(::RunningMeanVar{T}) where {T} = T
length(stat::RunningMeanVar) = stat.count
isempty(stat::RunningMeanVar) = (stat.count == 0)



"""
	push!(s::RunningMeanVar, x)

Append the value `x` to the collection represented by `s`.
"""
push!(stat::RunningMeanVar{T}, x) where {T} = push!(stat, convert(T, x))
function push!(stat::RunningMeanVar{T,R,S}, x::T) where {T,R,S}
	stat.count += 1

	if stat.count == 1
		stat.mean = x
		# stat.min = x
		# stat.max = x
		stat.sumsq = zero(S)
	else
		dx = x - stat.mean
		stat.mean += dx/stat.count
		stat.sumsq += (1.0-1.0/stat.count) * dot(dx, dx)
		# stat.min = min(x, stat.min)
		# stat.max = max(x, stat.max)
	end
	stat
end



"""
	mean(s::RunningMeanVar)

Mean of the collection represented by `s`. This has type `typeof(*(::T, Float64))`.
Returns `nothing` if `length(s) < 1`.

See also [`uncert`](@ref).
"""
mean(stat::RunningMeanVar) = stat.mean



"""
	var(s::RunningMeanVar; corrected = true)
	std(s::RunningMeanVar; corrected = true)

Variance and standard deviation of the collection represented by `s`.
This is generally real scalar, even when `T` is a container type.
Returns `nothing` if `length(s) < 2`.
	"""
var(stat::RunningMeanVar; corrected = true) = stat.count > 1 ? _var(stat, corrected) : nothing
std(stat::RunningMeanVar; corrected = true) = stat.count > 1 ? sqrt(_var(stat, corrected)) : nothing

# private
_var(stat::RunningMeanVar, corrected::Bool) = stat.sumsq/(stat.count - corrected)


"""
	uncert(s::RunningMeanVar)

Unbiased estimate of the statistical uncertainty of `mean(s)`. It is computed as
`sqrt(var(s)/length(s))` and is generally a real scalar.

The estimate is reliable only if the sample is large enough to be representative of
the distribution underlying the data.
"""
uncert(stat::RunningMeanVar) = stat.count > 1 ? sqrt.(_var(stat, true)/stat.count) : nothing



# minimum(stat::RunningStatistic) = stat.min
# maximum(stat::RunningStatistic) = stat.max



end # module RunningStatistics
