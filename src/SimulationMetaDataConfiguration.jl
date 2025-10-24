module SimulationMetaDataConfiguration

using TimerOutputs
using ProgressMeter
using Base: @kwdef

abstract type ShiftingMode end
struct ShiftingEnabled  <: ShiftingMode end
struct ShiftingDisabled <: ShiftingMode end

export SimulationMetaData, ShiftingEnabled, ShiftingDisabled


@kwdef mutable struct SimulationMetaData{Dimensions, FloatType <: AbstractFloat, S<:ShiftingMode}
    SimulationName::String
    SaveLocation::String
    HourGlass::TimerOutput                  = TimerOutput()
    Iteration::Int                          = 0
    OutputEach::FloatType                   = 0.02 #seconds
    OutputTimes::Union{FloatType,Vector{FloatType}} = OutputEach
    OutputIterationCounter::Int             = 0
    StepsTakenForLastOutput::Int            = 0
    CurrentTimeStep::FloatType              = 0
    TotalTime::FloatType                    = 0
    SimulationTime::FloatType               = 0
    IndexCounter::Int                       = 0
    ProgressSpecification::ProgressUnknown  =  ProgressUnknown(desc="Simulation time per output each:", spinner=true, showspeed=true) 
    VisualizeInParaview::Bool               = true
    ExportSingleVTKHDF::Bool                = true
    ExportGridCells::Bool                   = false
    OutputVariables::Vector{String}         = [
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
    OpenLogFile::Bool                       = true
    FlagOutputKernelValues::Bool            = false
    FlagLog::Bool                           = false
    shifting_mode::S                        = ShiftingDisabled()
    FlagSingleStepTimeStepping::Bool        = false
    ChunkMultiplier::Int                    = 1
    FlagMDBCSimple::Bool                    = false
end
end
