using Test
using SPHExampleGPU
using LinearAlgebra: dot, norm
using StaticArrays
using StructArrays
using Meshes

@testset "particle_struct_array builds particle fields" begin
    positions = [SVector(1.0, 2.0), SVector(3.0, 4.0)]
    particles = particle_struct_array(positions, 1000.0; GroupMarker = [1, 2])
    @test particles isa StructArray
    @test particles.Position == positions
    @test particles.Density == fill(1000.0, length(positions))
    @test particles.GroupMarker == [1, 2]
    @test !any(
        method -> method.module === SPHExampleGPU,
        methods(StructArray, Tuple{typeof(positions), Float64}),
    )
end

@testset "SPHGeometry holds input particles" begin
    mktempdir() do dir
        for d in (2, 3)
            positions = [SVector{d, Float64}(ntuple(k -> k + 0.123456789, d)),
                         SVector{d, Float64}(ntuple(k -> k + 0.234567891, d))]
            density = [1000.0, 1001.0]
            path = joinpath(dir, "particles$(d).csv")
            write_particle_csv(path, positions; density, first_id = 10)
            loaded = SPHGeometry{d, Float32}(CSVFile = path,
                GroupMarker = 1, Type = Fluid)
            direct = SPHGeometry{d, Float32}(Particles = StructArray((
                Position = positions, Density = density, ID = [11, 12])),
                GroupMarker = 1, Type = Fluid)
            @test loaded.Particles isa StructArray
            rm(path) # Allocation uses stored data, not the source file.
            meta = SimulationMetaData{d, Float32}(SimulationName = "geometry",
                SaveLocation = dir, GPUDoublePosition = true)
            a = AllocateDataStructures([loaded], meta)
            b = AllocateDataStructures([direct], meta)
            @test a == b
            @test a.Position == positions
            @test eltype(a.Position) == SVector{d, Float64}
            a.Position[1] = zero(eltype(a.Position))
            @test loaded.Particles.Position == positions
            @test AllocateDataStructures([loaded], meta) == b
            @test eltype(AllocateDataStructures([direct]).Position) ==
                SVector{d, Float32}
        end

        for d in (2, 3)
            positions = [SVector{d, Float32}(ntuple(k -> Float32(k) * 0.1f0, d)),
                         SVector{d, Float32}(ntuple(k -> Float32(k + 1) * 0.1f0, d)),
                         SVector{d, Float32}(ntuple(k -> Float32(k) * 0.1f0, d))]
            density = Float32[1000.1, 1000.2, 1000.3]
            path = joinpath(dir, "float32_particles$(d).csv")
            write_particle_csv(path, positions; density)
            loaded = SPHGeometry{d, Float32}(CSVFile = path,
                GroupMarker = 1, Type = Fluid)
            direct = SPHGeometry{d, Float32}(Particles = StructArray((
                Position = positions, Density = density, ID = [1, 2, 3])),
                GroupMarker = 1, Type = Fluid)
            meta = SimulationMetaData{d, Float32}(SimulationName = "geometry",
                SaveLocation = dir, GPUDoublePosition = true)
            loaded_particles = AllocateDataStructures([loaded], meta)
            direct_particles = AllocateDataStructures([direct], meta)
            @test loaded_particles == direct_particles
            @test loaded_particles.Position[1] == loaded_particles.Position[3]
        end

        regions = [ParticleRegion("wall", PolyArea([(0., 0.), (1., 0.),
                    (1., 1.), (0., 1.)]), Fixed)]
        region = only(sample_particles(regions, 0.5))
        overlapping = sample_particles([
            ParticleRegion("first", regions[1].geometry, Fixed),
            ParticleRegion("second", regions[1].geometry, Fluid),
        ], 0.5)
        @test !isempty(overlapping[1].positions)
        @test isempty(overlapping[2].positions)

        geometry = SPHGeometry{2, Float32}(region.positions;
            Density = 1000, GroupMarker = 1, Type = region.type)
        fluid = SPHGeometry{2, Float32}([SVector(2., 2.)];
            Density = [1001], GroupMarker = 2, Type = Fluid)
        particles = AllocateDataStructures([geometry, fluid])
        @test particles.ID == collect(1:length(particles))
        @test count(==(Fixed), particles.Type) == length(region.positions)
        @test particles.GroupMarker[end] == 2

        source = StructArray((Position = [SVector(1., 2.)], Density = [1000.],
            ID = [20], Velocity = [SVector(3., 4.)],
            GhostPoints = [SVector(1., 3.)], GhostNormals = [SVector(0., 1.)]))
        prescribed = SPHGeometry{2, Float32}(Particles = source,
            GroupMarker = 3, Type = Moving,
            Motion = MotionDetails{2, Float32}(; Velocity = 1, StartTime = 0,
                Duration = 1, Direction = SVector(1, 0)))
        mixed = AllocateDataStructures([geometry, prescribed])
        @test length(unique(mixed.ID)) == length(mixed)
        @test mixed.ID[end] > 20 # Automatically assigned IDs follow explicit IDs.
        k = findfirst(==(20), mixed.ID)
        @test mixed.Velocity[k] == SVector(3f0, 4f0)
        @test mixed.GhostPoints[k] == SVector(1f0, 3f0)
        @test mixed.GhostNormals[k] == SVector(0f0, 1f0)
        @test prescribed.Motion !== nothing
        @test_throws ArgumentError AllocateDataStructures([prescribed, prescribed])
        @test_throws ArgumentError SPHGeometry{2, Float32}(
            Particles = source, CSVFile = "unused", GroupMarker = 1, Type = Fluid)
        @test_throws DimensionMismatch SPHGeometry{3, Float32}(
            Particles = source, GroupMarker = 1, Type = Fluid)
        @test_throws DimensionMismatch SPHGeometry{2, Float32}(
            region.positions; Density = [1000], GroupMarker = 1, Type = Fixed)
        @test_throws ArgumentError SPHGeometry{2, Float32}(
            Particles = StructArray((Position = [SVector(0., 0.)],)),
            GroupMarker = 1, Type = Fluid)
    end
end

@testset "MDBC-ready boundary sampling" begin
    dp = 0.02
    centre = SVector(0.5, 0.5)
    boundary = sample_boundary(circle(centre, 0.08), dp)
    @test length(boundary.positions) < 128
    @test length(boundary.positions) > 3
    @test maximum(
        norm(boundary.positions[mod1(i + 1, length(boundary.positions))] -
             boundary.positions[i]) for i in eachindex(boundary.positions)
    ) <= dp / 2 + 1e-12
    @test all(isapprox(norm(n), dp / 2) for n in boundary.ghost_normals)
    @test all(boundary.ghost_points[i] ≈
              boundary.positions[i] + boundary.ghost_normals[i]
              for i in eachindex(boundary.positions))
    @test all(dot(boundary.ghost_normals[i], boundary.positions[i] - centre) > 0
              for i in eachindex(boundary.positions))

    tiny = sample_boundary(circle(centre, 1e-9), dp)
    @test length(tiny.positions) == 3
    @test maximum(
        norm(tiny.positions[mod1(i + 1, length(tiny.positions))] - tiny.positions[i])
        for i in eachindex(tiny.positions)
    ) < dp / 2
    @test all(all(isfinite, n) && norm(n) > 0 for n in tiny.ghost_normals)
    @test all(dot(tiny.ghost_normals[i], tiny.positions[i] - centre) > 0
              for i in eachindex(tiny.positions))

    square_boundary = sample_boundary(square((0, 0), 0.1), 0.2)
    corner = findfirst(==(SVector(0.0, 0.0)), square_boundary.positions)
    @test corner !== nothing
    @test square_boundary.ghost_normals[corner] / norm(square_boundary.ghost_normals[corner]) ≈
          SVector(-1, -1) / sqrt(2)

    inset = sample_boundary(square((0, 0), 1), 0.2;
                            spacing = 0.1, offset = 0.1, ghost_distance = 0.05)
    left_mid = findfirst(p -> isapprox(p[1], 0.1) && isapprox(p[2], 0.5),
                         inset.positions)
    @test left_mid !== nothing
    @test inset.ghost_normals[left_mid] ≈ SVector(-0.15, 0.0)
    @test inset.ghost_points[left_mid][1] ≈ -0.05

    narrow = sample_boundary(
        polygon([(-1, 0), (-0.5, -0.1), (0, -0.15), (0.5, -0.1), (1, 0),
                 (0.5, 0.1), (0, 0.15), (-0.5, 0.1)]),
        10.0,
    )
    @test length(narrow.positions) >= 3

    wall = SPHGeometry{2, Float32}(
        Particles = particle_struct_array(boundary.positions, 1000.0;
            GhostPoints = boundary.ghost_points,
            GhostNormals = boundary.ghost_normals),
        GroupMarker = 1,
        Type = Fixed,
    )
    particles = AllocateDataStructures([wall])
    @test particles.GhostPoints ≈ SVector{2, Float32}.(boundary.ghost_points)
    @test particles.GhostNormals ≈ SVector{2, Float32}.(boundary.ghost_normals)
    @test all(!iszero, particles.GhostPoints)
    @test_throws ArgumentError sample_boundary(circle(centre, 0.1), 0)
    @test_throws ArgumentError sample_boundary(circle(centre, 0.1), dp; spacing = 0)
    @test_throws ArgumentError sample_boundary(circle(centre, 0.1), dp; offset = -dp / 2)
    @test_throws ArgumentError sample_boundary(circle(centre, dp / 4), dp; offset = dp / 2)
    @test_throws ArgumentError sample_boundary(prism(circle(centre, 0.1), 0, 1), dp)
end

include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))
@testset "StillWedge generates geometry without CSV" begin
    constants = SimulationConstants{Float64}(dx = 0.02)
    geometry = still_wedge_2d_geometry(constants)
    @test length.(getproperty.(geometry, :Particles)) == [580, 2447]
    @test all(geom -> isempty(geom.CSVFile), geometry)
    particles = AllocateDataStructures(geometry)
    @test length(particles) == 3027
    @test all(isfinite, particles.Density)
end
