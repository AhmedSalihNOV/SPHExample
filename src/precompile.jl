using PrecompileTools
using StaticArrays
using StructArrays

@setup_workload begin
    D = 2
    T = Float64
    sc = SimulationConstants{T}()
    ker = SPHKernelInstance{D, T}(WendlandC2(); dx=sc.dx)
    pos = [SVector{D, T}(0, 0), SVector{D, T}(1, 0)]
    vel = [SVector{D, T}(0, 0), SVector{D, T}(0, 0)]
    acc = [SVector{D, T}(0, 0), SVector{D, T}(0, -9.81)]
    dens = fill(sc.ρ₀, 2)
    press = zeros(T, 2)
    particles = StructArray((Position=pos, Velocity=vel, Acceleration=acc))
end

@compile_workload begin
    Δt(pos, vel, acc, sc, ker)
    AllocateSupportDataStructures(particles.Position)
    Pressure!(press, dens, sc)
end
