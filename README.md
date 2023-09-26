# RunningStatistics.jl

Implements datatypes that accumulate statistics of data without storing the entire collection. Useful for streamed data.


`RunningStatistic{T}` is an abstract data type that accumulates statistics of serially input values of type `T`.  A `RunningStatistic` is used much like a collection: values are added by methods such as `push!`, and statistical properties are retrieved by methods such as `mean`. The full set of supported methods is listed below.

 Two concrete types are currently implemented:

* `RunningMeanVar{T} <: RunningStatistic{T}` accumulates the mean and variance of *single-variable values* of type `T`.  `T` can be any type for which algebraic operations and inner product are defined.

* `RunningMeanCov{T} <: RunningStatistic{Vector{T}}` accumulates a vector mean and covariance matrix of *multi-variable data* represented as vectors of values of type `T`.

The difference between the two is made clearer in the case of vector-valued data. A vector `x::Vector{T}` can be accumulated into either a `RunningMeanVar{Vector{T}}` or a `RunningMeanCov{T}`. The former treats `x` as a single vector-valued variable and computes a scalar-valued variance in terms of inner products.  In constrast, the latter treats each component of `x` as a separate variable and computes a covariance matrix with one element for each pair of vector components.

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
In many cases the sample mean is regarded as an estimate of the mean of an underlying population. In such contexts the quantity
```
uncert(s)		# 0.4
```
which is derived from the sample variance, provides an unbiased estimate of the statistical uncertainty in of sample mean.

The follow functions are implemented by `RunningMeanVar`:  `length`, `eltype`, `isempty`, `copy`, `push!`,
`append!`, `merge`, `mean`, `var`, `std`, and `uncert`. `RunningMeanCov` implements the same functions, except that it implements `cov` instead of `var`.


## Details

The mean and variance are computed using a generalization of the 1-pass method proposed by Welford (1962), which is simple, efficient, and moderately accurate.

For a `RunningMeanVar` with element type `T`, the following methods are needed to compute the mean and variance:
```
+(::T, ::T)
-(::T, ::T)
*(::Float64, ::T)
dot(::T, ::T)
```


The first three are used to compute the mean, which has type `R = typeof(*(::Float64, ::T))`.

The variance is computed using `dot` which is assumed to satisfy the properties of an inner product. The return type of `var`, `std`, and `uncert` is `S = typeof(real(dot(::R, ::R)))`, which is typically `Float64`.

The full type of the data structure is `RunningMeanVar{T,R,S}`.

