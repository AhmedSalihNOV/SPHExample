module SimulationMetaDataConfiguration

using TimerOutputs
using ProgressMeter

export SimulationMetaData, UpdateMetaData!, ShiftingMode, NoShifting, PlanarShifting,
       KernelOutputMode, NoKernelOutput, StoreKernelOutput,
       MDBCMode, NoMDBC, SimpleMDBC,
       LogMode, NoLog, StoreLog,
       TimeSteppingMode, SymplecticTimeStepping, SingleNeighborTimeStepping

# Mode types shared with the CPU package. They select code paths at compile
# time (as type parameters of `SimulationMetaData`) instead of run time flags,
# so that one case definition constructs against both packages.
abstract type ShiftingMode end
struct NoShifting     <: ShiftingMode end
struct PlanarShifting <: ShiftingMode end

abstract type KernelOutputMode end
struct NoKernelOutput    <: KernelOutputMode end
struct StoreKernelOutput <: KernelOutputMode end

abstract type MDBCMode end
struct NoMDBC     <: MDBCMode end
struct SimpleMDBC <: MDBCMode end

abstract type LogMode end
struct NoLog    <: LogMode end
struct StoreLog <: LogMode end

abstract type TimeSteppingMode end
struct SymplecticTimeStepping     <: TimeSteppingMode end
struct SingleNeighborTimeStepping <: TimeSteppingMode end

"""
    SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}(; SimulationName, SaveLocation, kwargs...)

Run time meta data of a simulation. Same type parameters, fields and defaults
as the CPU version plus a few GPU specific options (all prefixed `GPU`).

The mode parameters select code paths at compile time:

* `SMode <: ShiftingMode`: `NoShifting` (default) or `PlanarShifting`
* `KMode <: KernelOutputMode`: `NoKernelOutput` (default) or `StoreKernelOutput`
  (also store the kernel sum and kernel gradient sum of every particle)
* `BMode <: MDBCMode`: `NoMDBC` (default) or `SimpleMDBC` (requires
  `ParticleNormalsPath` in `RunSimulation`)
* `LMode <: LogMode`: `NoLog` (default) or `StoreLog`

Trailing mode parameters may be omitted, `SimulationMetaData{D, T}(...)`
selects all defaults. The time stepping scheme is not a type parameter; it is
passed to `RunSimulation` as `SimTimeStepping` and stored in the
`TimeSteppingMode` field.

Unlike the CPU version the keyword constructor converts `OutputTimes` (a number
or a vector of numbers) to `FloatType`, so `OutputTimes = 0.01` also works for
`FloatType = Float32`.
"""
mutable struct SimulationMetaData{Dimensions,
                                  FloatType <: AbstractFloat,
                                  SMode <: ShiftingMode,
                                  KMode <: KernelOutputMode,
                                  BMode <: MDBCMode,
                                  LMode <: LogMode}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput
    Iteration::Int
    OutputEach::FloatType
    OutputTimes::Union{FloatType, Vector{FloatType}}
    OutputIterationCounter::Int
    StepsTakenForLastOutput::Int
    CurrentTimeStep::FloatType
    TotalTime::FloatType
    SimulationTime::FloatType
    IndexCounter::Int
    ProgressSpecification::ProgressUnknown
    VisualizeInParaview::Bool
    ExportSingleVTKHDF::Bool
    ExportGridCells::Bool
    OutputVariables::Vector{String}
    OpenLogFile::Bool
    TimeSteppingMode::TimeSteppingMode
    # GPU specific options
    GPUSyncTimers::Bool          # synchronize after every phase so the timer output is meaningful
    GPUDeterministicSort::Bool   # sort particles inside each cell for bitwise reproducible runs
    GPUMaxCells::Int             # safety limit for the size of the neighbour grid
    GPUInteractionThreads::Int   # threads per block for the interaction kernels
    GPULanesPerParticle::Int     # warp lanes per particle in the gather kernels (0 = automatic)
    GPUBoundaryForces::Bool      # also evaluate the momentum equation for boundary particles (CPU parity)
    GPUAsyncOutput::Bool         # write output files on a Julia task while the GPU continues
end

const DEFAULT_OUTPUT_VARIABLES = [
    "ChunkID",
    "Kernel",
    "KernelGradient",
    "Density",
    "Pressure",
    "Velocity",
    "Acceleration",
    "BoundaryBool",
    "ID",
    "Type",
    "GroupMarker",
    "GhostPoints",
    "GhostNormals",
]

_output_times(::Type{T}, x::Real) where {T} = T(x)
_output_times(::Type{T}, x::AbstractVector) where {T} = Vector{T}(x)

function SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}(;
        SimulationName::String,
        SaveLocation::String,
        HourGlass::TimerOutput                  = TimerOutput(),
        Iteration::Int                          = 0,
        OutputEach                              = 0.02, # seconds
        OutputTimes                             = OutputEach,
        OutputIterationCounter::Int             = 0,
        StepsTakenForLastOutput::Int            = 0,
        CurrentTimeStep                         = 0,
        TotalTime                               = 0,
        SimulationTime                          = 0,
        IndexCounter::Int                       = 0,
        ProgressSpecification::ProgressUnknown  = ProgressUnknown(desc = "Simulation time per output each:", spinner = true, showspeed = true),
        VisualizeInParaview::Bool               = true,
        ExportSingleVTKHDF::Bool                = true,
        ExportGridCells::Bool                   = false,
        OutputVariables::Vector{String}         = copy(DEFAULT_OUTPUT_VARIABLES),
        OpenLogFile::Bool                       = true,
        TimeSteppingMode::TimeSteppingMode      = SingleNeighborTimeStepping(),
        GPUSyncTimers::Bool                     = false,
        GPUDeterministicSort::Bool              = true,
        GPUMaxCells::Int                        = 50_000_000,
        GPUInteractionThreads::Int              = 128,
        GPULanesPerParticle::Int                = 0,
        GPUBoundaryForces::Bool                 = true,
        GPUAsyncOutput::Bool                    = true,
    ) where {Dimensions, FloatType <: AbstractFloat, SMode <: ShiftingMode, KMode <: KernelOutputMode,
             BMode <: MDBCMode, LMode <: LogMode}
    return SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}(
        SimulationName, SaveLocation, HourGlass, Iteration,
        FloatType(OutputEach), _output_times(FloatType, OutputTimes),
        OutputIterationCounter, StepsTakenForLastOutput,
        FloatType(CurrentTimeStep), FloatType(TotalTime), FloatType(SimulationTime),
        IndexCounter, ProgressSpecification, VisualizeInParaview, ExportSingleVTKHDF, ExportGridCells,
        OutputVariables, OpenLogFile, TimeSteppingMode,
        GPUSyncTimers, GPUDeterministicSort, GPUMaxCells, GPUInteractionThreads, GPULanesPerParticle,
        GPUBoundaryForces, GPUAsyncOutput,
    )
end

# Trailing mode parameters default like in the CPU package.
SimulationMetaData{D,T,S,K,B}(; kwargs...) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,B<:MDBCMode} =
    SimulationMetaData{D,T,S,K,B,NoLog}(; kwargs...)
SimulationMetaData{D,T,S,K}(; kwargs...) where {D,T,S<:ShiftingMode,K<:KernelOutputMode} =
    SimulationMetaData{D,T,S,K,NoMDBC,NoLog}(; kwargs...)
SimulationMetaData{D,T,S}(; kwargs...) where {D,T,S<:ShiftingMode} =
    SimulationMetaData{D,T,S,NoKernelOutput,NoMDBC,NoLog}(; kwargs...)
SimulationMetaData{D,T}(; kwargs...) where {D,T} =
    SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(; kwargs...)

# Allow `meta.OutputTimes = 0.01` for Float32 meta data as well.
function Base.setproperty!(m::SimulationMetaData{D, T}, name::Symbol, x) where {D, T}
    if name === :OutputTimes
        return setfield!(m, name, _output_times(T, x))
    else
        return setfield!(m, name, convert(fieldtype(typeof(m), name), x))
    end
end

function UpdateMetaData!(SimMetaData, dt)
    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt
    return nothing
end

end
