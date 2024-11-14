# RunningStatistics.jl

Implements datatypes that accumulate statistics of data without storing the entire collection. Useful for streamed data.


`RunningStatistic{T}` is an abstract data type that accumulates statistics of serially input values of type `T`.  A `RunningStatistic` is used much like a collection: values are added by methods such as `push!`, and statistical properties are retrieved by methods such as `mean`. The full set of supported methods is listed below.

 Two concrete types are currently implemented:

* `RunningMeanVar{T} <: RunningStatistic{T}` accumulates the mean and variance of values of type `T`.  `T` can be any type for which algebraic operations and inner product are defined. Each value represents an observation of a single random variable.

* `RunningMeanCov{T} <: RunningStatistic{Vector{T}}` accumulates a mean and covariance matrix for values of type `Vector{T}`. Each vector represents a joint observation of multiple random variables.


## Usage

```
stat = RunningMeanVar{Float64}()
```
creates an object that maintains a running mean and variance for data of type `Float64`. Values are added to the (virtual) collection via `push!`:
```
push!(stat, 1.5)
push!(stat, 2.3)
```
The number of values in the collection is given by
```
length(stat)			# 2
```
The sample mean is obtained using
```
mean(stat)			# 1.9
```
while the sample variance and standard deviation can be obtained using
```
var(stat)			# 0.32
std(stat)			# 0.56568...
```
In many cases the sample mean is regarded as an estimate of the mean of an underlying population. In such contexts the variance of the mean
```
varmean(s)		# 0.16
```
provides an unbiased estimate of the variance of the sample mean as an estimator of the population mean. 

The follow functions are implemented by `RunningMeanVar`:  `length`, `eltype`, `isempty`, `copy`, `push!`,
`append!`, `empty!`, `merge`, `mean`, `var`, `std`, `relvar`, `varmean`, and `stdmean`. `RunningMeanCov` implements the same functions, except that it implements `cov` and `covmean` instead of `var` and `varmean`.


## Details

The mean, variance, and covariance are computed using a generalization of the 1-pass method proposed by Welford (1962), which is simple, efficient, and pretty accurate.

For a `RunningStatistic` with element type `T`, the following methods are needed:
```
+(::T, ::T)
-(::T, ::T)
*(::Float64, ::T)
dot(::T, ::T)
```
The first three are used to compute the mean. For a `RunningMeanVar{T}` the return type of `mean` is `R = typeof(*(::Float64, ::T))` whereas for a `RunningMeanCov{T}` it is `Vector{R}`.

The variance is computed using `dot` which is assumed to satisfy the properties of an inner product. For a `RunningMeanVar{T}` the return type of `var` and related quantities is `S = typeof(real(dot(::R, ::R)))`, which is typically `Float64`.  For a `RunningMeanCov{T}` the return type of `cov` and related quantities is `Matrix{typeof(dot(::R, ::R)}}`.

