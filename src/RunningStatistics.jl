"""
Datatypes to maintain running statistics (mean, min/max, variance, etc.) of data
without storing the entire collection. Useful for streamed data. 
"""
module RunningStatistics

using LinearAlgebra: dot, diag

import Base: push!, eltype, length, isempty, copy, merge, append!
import Statistics: mean, var, std, cov		# maximum, minimum

export RunningStatistic, RunningMeanVar, RunningMeanCov
export mean, var, cov, std, uncert  			# maximum, minimum


"""
	RunningStatistic{T}

A type that accumulates statistics of serially input values of type `T`.
It is used much like a collection and implements methods including
[`length`](@ref), [`eltype`](@ref), [`push!`](@ref), [`append!`](@ref), [`merge!`](@ref),
[`mean`](@ref), [`var`](@ref), [`std`](@ref), and [`uncert`](@ref). 
"""
abstract type RunningStatistic{T} end

err_not_impl(name::Symbol, args...) = error("$name is not implemented input types $(typeof.(args))")



"""
	push!(s::RunningStatistic, x)

Append the value `x` to the collection represented by `s`.
"""
push!(s::RunningStatistic, x...) = err_not_impl(:push!, s)



"""
	append!(s1::RunningStatistic, s2::RunningStatistic)

Incorporate the statistics of `s2` into `s1`. The result is as if the data
added to `s2` had been added to `s1`.  `s1` and `s2` must have the same type.

See also [`merge`](@ref).
"""
append!(s1::RunningStatistic, s2::RunningStatistic) = err_not_impl(:append!, (s1,s2))



	"""
	merge(s1::RunningStatistic, s2::RunningStatistic)

Creates a new `RunningMeanVar` that represents the combined data underlying `s1` and `s2`.
`s1` and `s2` must have the same type.

See also [`append!`](@ref).
"""
merge(s1::RunningStatistic, s2::RunningStatistic) = err_not_impl(:merge, (s1,s2))



"""
	mean(s::RunningStatistic)

Mean of the collection represented by `s`. This has type `typeof(*(::T, Float64))`.
Returns `nothing` if `length(s) < 1`.

See also [`uncert`](@ref).
"""
mean(s::RunningStatistic) = err_not_impl(:mean, s)





#-----------------------------------------------------
#
# RunningMeanVar
#


#
# Construction
#

"""
	RunningMeanVar{T}

A type that accumulates the mean and variance of serially input values of type `T`.
The type `T` must support addition/subtraction, scaling by a Float64, and two-argument `dot`.
"""
mutable struct RunningMeanVar{T,R,S} <: RunningStatistic{T}
	count::Int
	mean::Union{R, Nothing}
	uvar::Union{S, Nothing}		# uncorrected variance
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

push!(stat::RunningMeanVar{T}, x) where {T} = push!(stat, convert(T, x))
function push!(stat::RunningMeanVar{T,R,S}, x::T) where {T,R,S}
	stat.count += 1

	if stat.count == 1
		stat.mean = x
		stat.uvar = zero(S)
	else
		dx = x - stat.mean
		stat.mean += dx/stat.count
		stat.uvar = (1.0-1.0/stat.count) * (stat.uvar + real(dot(dx, dx))/stat.count)
	end
	stat
end



function append!(stat1::RunningMeanVar{T,R,S}, stat2::RunningMeanVar{T,R,S}) where {T,R,S}
	N1 = stat1.count
	N2 = stat2.count
	if N1 == 0
		stat1.count = stat2.count
		stat1.mean = stat2.mean
		stat1.uvar = stat2.uvar
	elseif N2 > 0
		f1 = N1/(N1+N2)
		f2 = N2/(N1+N2)
		y = stat1.mean - stat2.mean
		stat1.count = N1 + N2
		stat1.mean = f1 * stat1.mean + f2 * stat2.mean
		stat1.uvar = f1 * stat1.uvar + f2 * stat2.uvar + f1*f2 * dot(y,y)
	end
	return stat1
end



merge(stat1::RunningMeanVar{T,R,S}, stat2::RunningMeanVar{T,R,S}) where {T,R,S} = append!(copy(stat1), stat2)


#
# Statistical quantities
#

mean(stat::RunningMeanVar) = stat.mean


"""
	var(s::RunningMeanVar; corrected = true)

Variance of the collection represented by `s`.  This is generally a real scalar,
even when `T` is a container type.  Returns `nothing` if `length(s) < 2`.
"""
var(stat::RunningMeanVar; corrected = true) = stat.count > 1 ? _var(stat, corrected) : nothing



"""
	std(s::RunningMeanVar; corrected = true)

Standard deviation of the collection represented by `s`.  This is generally a real scalar,
even when `T` is a container type.  Returns `nothing` if `length(s) < 2`.
"""
std(stat::RunningMeanVar; corrected = true) = stat.count > 1 ? sqrt(_var(stat, corrected)) : nothing



"""
	uncert(s::RunningMeanVar)

Unbiased estimate of the statistical uncertainty of `mean(s)`, given by
`sqrt(var(s)/length(s))`.

The estimate is reliable only if the sample is large enough to be representative of
the distribution underlying the data.
"""
uncert(stat::RunningMeanVar) = stat.count > 1 ? sqrt.(_var(stat, true)/stat.count) : nothing


# private
_var(stat::RunningMeanVar, corrected::Bool) = corrected ? stat.uvar/(1.0-1.0/stat.count) : stat.uvar





#-----------------------------------------------------
#
# RunningMeanCov
#

#
# Construction
#

"""
	RunningMeanCov{T} <: RunningStatistic{Vector{T}}

A type that accumulates a vector mean and covariance matrix of multivariate data.
Each datum is a vector whose elements are values of distinct variables of type `T`.
"""
mutable struct RunningMeanCov{T,R,S} <: RunningStatistic{Vector{T}}
	count::Int
	mean::Union{Vector{R}, Nothing}
	ucov::Union{Matrix{S}, Nothing}		# uncorrected covariance matrix
	# min::Union{T, Nothing}
	# max::Union{T, Nothing}
end


function RunningMeanCov{T}() where {T}
	R = Base.return_types(*, (T, Float64))[1]
	S = Base.return_types(dot, (R, R))[1]		# or promote_type(eltype(T), Float64)
	RunningMeanCov{T,R,S}(0, nothing, nothing)
end


copy(s::RunningMeanCov{T,R,S}) where {T,R,S} = RunningMeanCov{T,R,S}(s.count, s.mean, s.ucov)


#
# Basic properties
#

eltype(::RunningMeanCov{T}) where {T} = Vector{T}
length(stat::RunningMeanCov) = stat.count
isempty(stat::RunningMeanCov) = (stat.count == 0)


#
# Incorporating data
#

push!(stat::RunningMeanCov{T}, x::NTuple) where {T} = push!(stat, collect(convert.(T, x)))
push!(stat::RunningMeanCov{T}, x::Vector) where {T} = push!(stat, convert.(T, x))
function push!(stat::RunningMeanCov{T,R,S}, x::Vector{T}) where {T,R,S}
	stat.count += 1

	if stat.count == 1
		n = length(x)
		stat.mean = x
		stat.ucov = zeros(S, (n,n))
		# stat.min = x
		# stat.max = x
	else
		n = length(x)
		length(stat.mean) == n || error("Input vector x has the wrong number of elements")

		dx = x - stat.mean
		stat.mean += dx/stat.count
		for i = 1:n
			for j = i:n
				c = (1.0-1.0/stat.count) * (stat.ucov[i,j] + dot(dx[j], dx[i])/stat.count)
				stat.ucov[i,j] = c
				stat.ucov[j,i] = c'
			end
		end 
	end
	stat
end



function append!(stat1::RunningMeanCov{T,R,S}, stat2::RunningMeanCov{T,R,S}) where {T,R,S}
	n = length(stat1.mean)
	length(stat2.mean) == n || error("Inputs stat1 and stat2 have data of different size")

	N1 = stat1.count
	N2 = stat2.count
	if N1 == 0
		stat1.count = stat2.count
		stat1.mean = stat2.mean
		stat1.uvar = stat2.uvar
	elseif N2 > 0
		f1 = N1/(N1+N2)
		f2 = N2/(N1+N2)
		y = stat1.mean - stat2.mean
		stat1.count = N1 + N2
		stat1.mean = f1 * stat1.mean + f2 * stat2.mean
		for i = 1:n
			for j = i:n
				c = f1 * stat1.ucov[i,j] + f2 * stat2.ucov[i,j] + f1*f2 * dot(y[j],y[i])
				stat1.uvar[i,j] = c
				stat1.uvar[j,i] = c'
			end
		end
	end
	return stat1
end



merge(stat1::RunningMeanCov{T,R,S}, stat2::RunningMeanCov{T,R,S}) where {T,R,S} = append!(copy(stat1), stat2)


#
# Statistical quantities
#

mean(stat::RunningMeanCov) = stat.mean


"""
	cov(s::RunningMeanCov)

Returns the covariance matrix for the multivariate data represented by `s`. 
"""
cov(stat::RunningMeanCov; corrected = true) = stat.count > 1 ? _cov(stat, corrected) : nothing


"""
	std(s::RunningMeanCov; corrected = true)

Standard deviation of the collection represented by `s`; square root of the
covariance matrix.  Returns `nothing` if `length(s) < 2`.
"""
std(stat::RunningMeanCov; corrected = true) = stat.count > 1 ? sqrt(_cov(stat, corrected)) : nothing



"""
	uncert(s::RunningMeanVar)

Unbiased estimate of the statistical uncertainty of `mean(s)`, given by
`sqrt(cov(s)/length(s))`.

The estimate is reliable only if the sample is large enough to be representative of
the distribution underlying the data.
"""
uncert(stat::RunningMeanCov) = stat.count > 1 ? sqrt(_cov(stat, true)/stat.count) : nothing


# private
_cov(stat::RunningMeanCov, corrected::Bool) = corrected ? stat.ucov/(1.0-1.0/stat.count) : stat.ucov



end # module RunningStatistics
