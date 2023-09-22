"""
Datatypes to maintain running statistics (mean, min/max, variance, etc.) of data
without storing the entire collection. Useful for streamed data. 
"""
module RunningStatistics

using LinearAlgebra: dot

import Base: push!, eltype, length, isempty, copy, merge, append!
import Statistics: mean, var, std		# maximum, minimum

export RunningMeanVar
export mean, var, std, uncert  			# maximum, minimum


#
# Construction
#

"""
	RunningMeanVar{T}()

An object that keeps a running mean and variance of serially input data
of type `T`.  It is used much like a collection and implements methods including
[`length`](@ref), [`eltype`](@ref), [`push!`](@ref), [`append!`](@ref), [`merge!`](@ref),
[`mean`](@ref), [`var`](@ref), [`std`](@ref), and [`uncert`](@ref). 

The type `T` must support addition/subtraction, scaling by a Float64, and
two-argument `dot`.
"""
mutable struct RunningMeanVar{T,R,S}
	count::Int
	mean::Union{R, Nothing}
	uvar::Union{S, Nothing}		# uncorrected variance
	# min::Union{T, Nothing}
	# max::Union{T, Nothing}
end


function RunningMeanVar{T}() where {T}
	R = Base.return_types(*, (T, Float64))[1]
	S = Base.return_types(real∘dot, (R, R))[1]		# or promote_type(eltype(T), Float64)
	RunningMeanVar{T,R,S}(0, nothing, nothing)
end


copy(s::RunningMeanVar{T,R,S}) where {T,R,S} = RunningMeanVar{T,R,S}(s.count, s.mean, s.uvar)


#
# Basic properties
#

eltype(::RunningMeanVar{T}) where {T} = T
length(stat::RunningMeanVar) = stat.count
isempty(stat::RunningMeanVar) = (stat.count == 0)


#
# Incorporating data
#

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
		stat.uvar = zero(S)
	else
		dx = x - stat.mean
		stat.mean += dx/stat.count
		stat.uvar = (1.0-1.0/stat.count) * (stat.uvar + real(dot(dx, dx))/stat.count)
		# stat.min = min(x, stat.min)
		# stat.max = max(x, stat.max)
	end
	stat
end



"""
	append!(s1::RunningMeanVar, s2::RunningMeanVar)

Update `s1` to reflect the combined data underlying `s1` and `s2`. The result is as if
the data added to `s2` had been added to `s1`.  `s1` and `s2` must have the same type.

See also [`merge`](@ref).
"""
function append!(stat1::RunningMeanVar{T,R,S}, stat2::RunningMeanVar{T,R,S}) where {T,R,S}
	n1 = stat1.count
	n2 = stat2.count
	if n1 == 0
		stat1.count = stat2.count
		stat1.mean = stat2.mean
		stat1.uvar = stat2.uvar
	elseif n2 > 0
		f1 = n1/(n1+n2)
		f2 = n2/(n1+n2)
		y = stat1.mean - stat2.mean
		stat1.count = n1 + n2
		stat1.mean = f1 * stat1.mean + f2 * stat2.mean
		stat1.uvar = f1 * stat1.uvar + f2 * stat2.uvar + f1*f2 * dot(y,y)
	end
	return stat1
end



"""
	merge(s1::RunningMeanVar, s2::RunningMeanVar)

Creates a new `RunningMeanVar` that represents the combined data underlying `s1` and `s2`.
`s1` and `s2` must have the same type.

See also [`append!`](@ref).
"""
merge(stat1::RunningMeanVar{T,R,S}, stat2::RunningMeanVar{T,R,S}) where {T,R,S} = append!(copy(stat1), stat2)


#
# Statistical quantities
#

"""
	mean(s::RunningMeanVar)

Mean of the collection represented by `s`. This has type `typeof(*(::T, Float64))`.
Returns `nothing` if `length(s) < 1`.

See also [`uncert`](@ref).
"""
mean(stat::RunningMeanVar) = stat.mean



"""
	var(s::RunningMeanVar; corrected = true)

Variance of the collection represented by `s`.  This is generally a real scalar,
even when `T` is a container type.  Returns `nothing` if `length(s) < 2`.
"""
var(stat::RunningMeanVar; corrected = true) = stat.count > 1 ? _var(stat, corrected) : nothing

# private
_var(stat::RunningMeanVar, corrected::Bool) = corrected ? stat.uvar/(1.0-1.0/stat.count) : stat.uvar


"""
	std(s::RunningMeanVar; corrected = true)

Standard deviation of the collection represented by `s`.  This is generally a real scalar,
even when `T` is a container type.  Returns `nothing` if `length(s) < 2`.
"""
std(stat::RunningMeanVar; corrected = true) = stat.count > 1 ? sqrt(_var(stat, corrected)) : nothing



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
