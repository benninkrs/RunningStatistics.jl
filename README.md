# RunningStatistics.jl

Datatypes to maintain running statistics (mean, min/max, variance, etc.) of data without storing the entire collection. Useful for streamed data.

## Usage

```
stat = RunningMeanVar{Float64}()
```
creates an object that maintains a running mean and variance for data of type `Float64`.  This object can be used much like a collection. Values are added to the (virtual) collection via `push!`:
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

The follow methods are supported for `RunningMeanVar`:  `length`, `eltype`, `isempty`, `copy`, `push!`,
`append!`, `merge`, `mean`, `var`, `std`, and `uncert`.


In general, `RunningMeanVar{T}()` tracks the mean and variance for data of type `T` (see below for requirements on `T`.)  Notably, `var`, `std`, and `uncert` generally return real scalars corresponding to distances in `T` space, even when `T` is a vector type.

If you have data of type `Vector{T}` and wish to track the mean and variance of *each component independently*, do not create a `RunningMeanVar{Vector{T}}`; instead create a `Vector{RunningMeanVar{T}}` and broadcast `push!`, `mean`, etc.


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

