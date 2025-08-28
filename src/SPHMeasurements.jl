module SPHMeasurements

using StaticArrays
using LinearAlgebra
using HDF5
using ..SPHKernels
using ..AuxiliaryFunctions: to_3d
using ..ProduceHDFVTK: SaveVTKHDF

export MeasurementPoint, MeasurementLine, MeasurementGrid,
       measurement_points, interpolate_field,
       SimMeasurement, SimMeasurements, perform_measurements!,
       save_measurement_vtk

abstract type MeasurementRegion end

"""
    MeasurementPoint(point)

Represent a single measurement location.
"""
struct MeasurementPoint{D,T} <: MeasurementRegion
    point::SVector{D,T}
end

"""
    MeasurementLine(line_start, line_end, num_points)

A line segment discretized into `num_points` points including the
endpoints.
"""
struct MeasurementLine{D,T} <: MeasurementRegion
    line_start::SVector{D,T}
    line_end::SVector{D,T}
    num_points::Int
end

"""
    MeasurementGrid(origin, spacing, dims)

Axis-aligned grid beginning at `origin` with spacing `spacing` and
`dims` points in each direction.
"""
struct MeasurementGrid{D,T} <: MeasurementRegion
    origin::SVector{D,T}
    spacing::SVector{D,T}
    dims::NTuple{D,Int}
end

"""
    measurement_points(region)

Return the discretized measurement points for `region`.
"""
measurement_points(region::MeasurementPoint) = [region.point]

function measurement_points(region::MeasurementLine{D,T}) where {D,T}
    pts = Vector{SVector{D,T}}(undef, region.num_points)
    
    if region.num_points == 1
        pts[1] = region.line_start
        return pts
    end
    
    for i in 1:region.num_points
        t = (i - 1) / (region.num_points - 1)
        pts[i] = region.line_start + t * (region.line_end - region.line_start)
    end

    return pts
end

using StaticArrays

function measurement_points(region::MeasurementGrid{D,T}) where {D,T}
    dims    = region.dims
    origin  = region.origin        # SVector{D,T}
    spacing = region.spacing       # SVector{D,T}

    n   = prod(dims)
    pts = Vector{SVector{D,T}}(undef, n)

    @inbounds for (i, I) in enumerate(CartesianIndices(dims))
        # Build offset without Tuple(I) and convert to T once
        offset = SVector{D,T}(ntuple(j -> T(I[j] - 1), Val(D)))
        pts[i] = origin + spacing .* offset
    end
    return pts
end

"""
    interpolate_field(points, field, particles, constants, kernel)

Interpolate scalar `field` defined on `particles` at `points` using
`kernel`. The particle mass is taken from `constants.m₀` and density
from `particles.Density`.
"""
function interpolate_field(points, field, particles, constants, kernel::SPHKernelInstance)
    m₀ = constants.m₀
    ρ = particles.Density
    x = particles.Position
    
    # Pre-allocate with correct type
    T = promote_type(eltype(field), typeof(m₀))
    result = Vector{T}(undef, length(points))
    
    for (pi, p) in enumerate(points)
        s = zero(T)
        for i in eachindex(x)
            dx = p - x[i]
            r2 = dot(dx, dx)
            r2 > kernel.H² && continue
            
            q = sqrt(r2) * kernel.h⁻¹
            w = Wᵢⱼ(kernel, q)
            s += field[i] * w * m₀ / ρ[i]
        end
        result[pi] = s
    end
    return result
end

"""
    interpolate_field(region, field, particles, constants, kernel)

Interpolate `field` at the discretized points of `region`.
"""
function interpolate_field(region::MeasurementRegion, field, particles, constants,
                           kernel::SPHKernelInstance)
    points = measurement_points(region)
    return interpolate_field(points, field, particles, constants, kernel)
end

"""
    SimMeasurement(region, field)

Container that records interpolated values of `field` over `region` for
specified simulation times. Results are stored in the `data` dictionary
with the time value as key.
"""
struct SimMeasurement{R<:MeasurementRegion}
    region::R
    field::Symbol
    data::Dict{Float64,Any}
    SimMeasurement(region::R, field::Symbol) where {R<:MeasurementRegion} =
        new{R}(region, field, Dict{Float64,Any}())
end

"""
    SimMeasurements(measurements, times)

Holds a collection of `measurements` to be sampled at the simulation
`times` provided.
"""
struct SimMeasurements
    measurements::Vector{SimMeasurement}
    times::Vector{Float64}
end

"""
    perform_measurements!(sim_measurements, time, particles, constants, kernel)

If `time` is approximately equal to one of `sim_measurements.times`,
interpolate the requested fields and store them in each measurement's
`data` dictionary.
"""
function perform_measurements!(sm::SimMeasurements, time, particles, constants,
                               kernel::SPHKernelInstance)
    any(t -> isapprox(time, t; atol = 1e-12), sm.times) || return
    for m in sm.measurements
        field = getproperty(particles, m.field)
        m.data[time] = interpolate_field(m.region, field, particles, constants, kernel)
    end
    return nothing
end

"""
    save_measurement_vtk(measurement, time, filepath)

Write measurement `data` at `time` to `filepath` in VTKHDF format.
"""
function save_measurement_vtk(measurement::SimMeasurement, time, filepath)
    haskey(measurement.data, time) ||
        error("no data stored for time $(time)")
    points = to_3d(measurement_points(measurement.region))
    values = measurement.data[time]
    fids = Vector{HDF5.File}(undef, 1)
    SaveVTKHDF(fids, 1, filepath, points, [String(measurement.field)], values)
    close(fids[1])
    return nothing
end

end # module

