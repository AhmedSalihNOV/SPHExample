module SPHMeasurements

using Parameters

export SimulationMeasurements

"""
    SimulationMeasurements{T}

Container to store measurement data collected during a simulation.
"""
@with_kw mutable struct SimulationMeasurements{T<:AbstractFloat}
    times::Vector{T} = T[]
end

end
