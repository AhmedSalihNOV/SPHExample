using Test
using SPHExample

const ThreadPool = SPHExample.SPHThreadPool

@testset "persistent worker pool" begin
    @testset "every index is visited exactly once" begin
        # Cover empty input, both sides of a batch boundary, partial final
        # batches, and enough batches to exercise every worker several times.
        for Count in (0, 1, 63, 64, 65, 127, 128, 129, 513, 10_000)
            Visits = [Threads.Atomic{Int}(0) for _ in 1:Count]
            Values = zeros(Int, Count)
            Result = ThreadPool.ForEachIndex!(1:Count, 64) do Index
                Threads.atomic_add!(Visits[Index], 1)
                Values[Index] = Index^2
            end
            @test Result === nothing
            @test all(Visit -> Visit[] == 1, Visits)
            @test Values == (1:Count).^2
        end
    end

    @testset "regions can follow one another" begin
        # Workers park on a generation counter, so a region must not be able to
        # observe the ticket of the region before it.
        Values = zeros(Int, 1024)
        for Round in 1:50
            ThreadPool.ForEachIndex!(1:1024, 64) do Index
                Values[Index] = Round * Index
            end
            @test Values == Round .* (1:1024)
        end
    end

    @testset "the pool never oversubscribes a core" begin
        # Workers spin between regions, so a second worker on a core that
        # already has one costs more than it contributes.
        Workers = ThreadPool.WorkerCount()
        @test 1 <= Workers <= Threads.nthreads(:default)
        @test Workers <= ThreadPool.PhysicalCoreCount()
        @test ThreadPool.PhysicalCoreCount() >= 1
    end

    @testset "every worker index is used exactly once" begin
        Workers = ThreadPool.WorkerCount()
        Visits = [Threads.Atomic{Int}(0) for _ in 1:Workers]
        Reported = [Threads.Atomic{Int}(0) for _ in 1:Workers]
        ThreadPool.ForEachWorker!() do WorkerIndex, Count
            Threads.atomic_add!(Visits[WorkerIndex], 1)
            Reported[WorkerIndex][] = Count
        end
        @test all(Visit -> Visit[] == 1, Visits)
        @test all(Count -> Count[] == Workers, Reported)
    end

    @testset "a nested region runs serially instead of deadlocking" begin
        # The inner region would otherwise reset the cursor the outer loop is
        # still reading from, and neither would ever finish.
        Outer = zeros(Int, 256)
        Inner = zeros(Int, 256)
        ThreadPool.ForEachIndex!(1:256, 8) do Index
            Outer[Index] = Index
            ThreadPool.ForEachIndex!(1:256, 8) do Nested
                Inner[Nested] = Nested
            end
        end
        @test Outer == collect(1:256)
        @test Inner == collect(1:256)
    end

    @testset "worker failures reach the caller" begin
        Failure = try
            ThreadPool.ForEachIndex!(1:1029, 64) do Index
                Index == 65 && error("intentional pool worker failure")
            end
            nothing
        catch Exception
            Exception
        end
        @test Failure isa Exception
        @test occursin("intentional pool worker failure", sprint(showerror, Failure))

        # A failed region must not leave the pool holding its region guard.
        Values = zeros(Int, 513)
        ThreadPool.ForEachIndex!(1:513, 64) do Index
            Values[Index] = Index
        end
        @test Values == collect(1:513)
    end

    @testset "worker bodies defined after the pool was created still run" begin
        # The workers outlive the world age they were spawned in, so a body
        # whose method appears later is only callable through invokelatest.
        ThreadPool.ForEachIndex!(Index -> nothing, 1:512, 64)

        @eval LateWorldValues = zeros(Int, 512)
        @eval LateWorldBody(Index) = (LateWorldValues[Index] = Index^2)
        ThreadPool.ForEachIndex!(LateWorldBody, 1:512, 64)
        @test LateWorldValues == (1:512).^2
    end
end
