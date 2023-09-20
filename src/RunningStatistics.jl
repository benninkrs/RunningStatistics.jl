"""
Datatype to maintain running statistics (mean, min/max, variance, etc.) of data.
Useful for streamed data or when you don't want to stroe all the data in memory. 
"""
module RunningStatistics

import Base: push!
import Statistics: mean, var, std, maximum, minimum

export RunningStatistic
export mean, var, std, maximum, minimum, uncert

"""
	RunningStatistic{T}()
	RunnningStatistic(x::T)

An object that maintins basic statistics of multiple data of type T.
T can be any type that implements addition/subtraction, min/max, scaling, and
(elementise) `sqrt`.

A `RunningStatistic` is used much like a collection, and supports the following methods:
`push!`, `length`, `eltype`, `mean`, `var`, `std`, `min`, `max`.  The new method `uncert`
provides an estimate of the uncertainty of the mean.
"""
mutable struct RunningStatistic{T}
	count::Int
	min::Union{T, Nothing}
	max::Union{T, Nothing}
	mean::Union{T, Nothing}
	sumsq::Union{T, Nothing}

	# Create with explicit type
	function RunningStatistic{T}() where {T}
		new(0, nothing, nothing, nothing, nothing)
	end

	# Create from template object
	function RunningStatistic(x::T) where {T}
		new{T}(0, nothing, nothing, nothing, nothing)
	end

end


eltype(::RunningStatistic{T}) where {T} = T
length(stat::RunningStatistic) = stat.count

mean(stat::RunningStatistic) = stat.mean
minimum(stat::RunningStatistic) = stat.min
maximum(stat::RunningStatistic) = stat.max


raw_var(stat::RunningStatistic, corrected::Bool) = stat.sumsq/(stat.count - corrected)

var(stat::RunningStatistic; corrected = true) = stat.count > 1 ? raw_var(stat, corrected) : nothing
std(stat::RunningStatistic; corrected = true) = stat.count > 1 ? sqrt.(raw_var(stat, corrected)) : nothing

"""
	uncert(s::RunningStatistic)

Estimated statistical uncertainty of `mean(s)``;  equivalent to `sqrt(var(s)/length(s))`.
Like `var` and `std`, `uncert` should not be relied upon until the collection is large enough
to fairly represent the population.
"""
uncert(stat::RunningStatistic) = stat.count > 1 ? sqrt.(raw_var(stat, true)/stat.count) : nothing

"""
	push!(s::RunningStatistic, x)

Append the value `x` to the collection represented by `s`.
"""
push!(stat::RunningStatistic{T}, x) where {T} = push!(stat, convert(T, x))
function push!(stat::RunningStatistic{T}, x::T) where {T}
	stat.count += 1

	if stat.count == 1
		stat.mean = x
		stat.min = x
		stat.max = x
		stat.sumsq = zero(x)
	else
		mean_ = stat.mean
		stat.mean += (x - mean_)/stat.count
		stat.sumsq += (x - stat.mean) .* (x - mean_) 
		stat.min = min(x, stat.min)
		stat.max = max(x, stat.max)
	end
	stat
end



# mean(stat::RunningStatistic) = stat.count > 0 ? stat.mean : nothing
# var(stat::RunningStatistic; corrected = true) = stat.count > 1 ? raw_var(stat, corrected) : nothing
# std(stat::RunningStatistic; corrected = true) = stat.count > 1 ? sqrt(raw_var(stat, corrected)) : nothing
# minimum(stat::RunningStatistic) = stat.count > 0 ? stat.min : nothing
# maximum(stat::RunningStatistic) = stat.count > 0 ? stat.max : nothing
# uncertainty(stat::RunningStatistic) = stat.count > 1 ? sqrt(raw_var(stat, true)/stat.count) : nothing





end # module RunningStatistics
