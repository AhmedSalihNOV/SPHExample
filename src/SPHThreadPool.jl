module SPHThreadPool

export ForEachIndex!, ForEachWorker!, WorkerCount

using Base.Threads: nthreads, Atomic, atomic_add!

    # Every parallel region of an SPH step used to spawn and join a fresh task
    # set. At a few thousand particles the step is short enough that the spawn
    # and join cost dominates the physics, which is why the small 2D cases run
    # better on half the machine. The workers here are created once and park
    # between regions instead, so opening one costs a store per worker.
    #
    # Region bodies take (WorkerIndex, WorkerCount) and the coordinator runs as
    # worker 1, so the caller's Bumper buffers stay alive for the whole region:
    # nothing returns until every worker has published its completion.
    #
    # The workers live for as long as the process does; there is no shutdown
    # path, because a pool that outlives its last region costs one parked task
    # per core and nothing else.

    # Spin iterations a worker performs before parking. Long enough to cover the
    # gap between two regions of the same step, short enough that the threads go
    # back to the scheduler whenever the solver pauses, which the VTKHDF writer
    # task in ProduceHDFVTK needs in order to make progress.
    const WorkerSpinBudget = 2_000

    # The coordinator yields this often while waiting so that it cannot
    # monopolise its own thread against anything queued behind it.
    const CoordinatorSpinBudget = 4_000

    @inline CpuPause() = ccall(:jl_cpu_pause, Cvoid, ())

    # One counter per cache line. Workers poll their own ticket and publish to
    # their own completion slot, so opening a region is a handful of independent
    # stores rather than every worker contending for one shared line.
    #
    # Tickets and completion slots are only ever loaded and stored, so a padded
    # field is all they need. The batch cursor is the one counter every worker
    # increments, and `@atomic Field += n` lowers to a cmpxchg loop, so it uses
    # `Atomic{Int}` to get a real fetch-and-add.
    mutable struct PaddedCounter
        @atomic Value::Int
        Padding::NTuple{15, Int}
    end

    PaddedCounter() = PaddedCounter(0, ntuple(_ -> 0, Val(15)))

    mutable struct WorkerPool
        const Tasks::Vector{Task}
        const Count::Int
        const Tickets::Vector{PaddedCounter}
        const Completed::Vector{PaddedCounter}
        const Sleepers::Vector{PaddedCounter}
        const Barriers::Vector{Base.Threads.Condition}
        const Cursor::Atomic{Int}
        const Failures::Vector{Any}
        @atomic Job::Any
        @atomic Generation::Int
        @atomic Active::Bool
    end

    # Workers spin between regions so that opening the next one costs a store
    # rather than a scheduler round trip, and that only pays while each worker
    # has a core to itself. Measured on a 16-core / 24-thread hybrid laptop with
    # StillWedge2D mDBC at `-t 24,0`: 457 us/step with a worker on every thread,
    # 244 us/step with the pool held to the 16 physical cores, against 359
    # us/step for the `@threads` path this replaces. The thread count stays the
    # user's choice; the pool only declines to put a second spinning worker on a
    # core that already has one.
    function CountPhysicalCores()
        try
            if Sys.iswindows()
                # GetLogicalProcessorInformationEx(RelationProcessorCore, ...)
                # returns one variable-length record per physical core.
                RelationProcessorCore = UInt32(0)
                Length = Ref{UInt32}(0)
                ccall((:GetLogicalProcessorInformationEx, "kernel32"), Bool,
                      (UInt32, Ptr{Cvoid}, Ptr{UInt32}), RelationProcessorCore, C_NULL, Length)
                Length[] == 0 && return 0
                Buffer = Vector{UInt8}(undef, Length[])
                ccall((:GetLogicalProcessorInformationEx, "kernel32"), Bool,
                      (UInt32, Ptr{UInt8}, Ptr{UInt32}), RelationProcessorCore, Buffer, Length) || return 0
                Offset = 0
                Cores = 0
                while Offset + 8 <= Int(Length[])
                    # Record layout is {DWORD Relationship; DWORD Size; ...}.
                    Size = Int(only(reinterpret(UInt32, view(Buffer, Offset+5:Offset+8))))
                    Size <= 0 && break
                    Cores += 1
                    Offset += Size
                end
                return Cores
            elseif Sys.islinux()
                # One entry per core, listing the hardware threads sharing it.
                Siblings = Set{String}()
                for Entry in readdir("/sys/devices/system/cpu")
                    startswith(Entry, "cpu") || continue
                    Path = "/sys/devices/system/cpu/$Entry/topology/thread_siblings_list"
                    isfile(Path) && push!(Siblings, strip(read(Path, String)))
                end
                return length(Siblings)
            elseif Sys.isapple()
                return parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
            end
        catch
            # Topology is an optimisation only, so any failure just means the
            # pool sizes itself from the thread count instead.
        end
        return 0
    end

    const PhysicalCores = Ref(0)

    # Resolved on first use rather than at load time: a value baked in during
    # precompilation would describe the build machine, not this one.
    function PhysicalCoreCount()
        Cores = PhysicalCores[]
        Cores == 0 || return Cores
        Cores = CountPhysicalCores()
        Cores > 0 || (Cores = Sys.CPU_THREADS)
        PhysicalCores[] = Cores
        return Cores
    end

    """
        WorkerCount()

    Number of workers a parallel region is split across, counting the
    coordinator. Returns `1` whenever regions run serially, so callers that size
    per-worker storage never disagree with the pool.
    """
    @inline function WorkerCount()
        Threading = nthreads(:default)
        Threading > 1 || return 1
        # No tasks during precompilation: they would not survive into the image.
        ccall(:jl_generating_output, Cint, ()) == 1 && return 1
        return min(Threading, PhysicalCoreCount())
    end

    function SpawnPinnedWorker(Body::F, GlobalThreadId::Int) where {F}
        Worker = Task(Body)
        Worker.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), Worker, GlobalThreadId - 1)
        schedule(Worker)
        return Worker
    end

    # Parking is per worker rather than on one shared condition: a single
    # condition would have every wake-up broadcast to all workers at once, and
    # they would then file through the same lock one at a time to re-read their
    # ticket. The `Sleeping` flag lets the coordinator skip the lock entirely
    # for any worker that is still spinning, which is the common case between
    # two regions of the same step.
    function AwaitTicket(Ticket::PaddedCounter, Sleeping::PaddedCounter,
                         Barrier::Base.Threads.Condition, Seen::Int)
        for _ in 1:WorkerSpinBudget
            Current = @atomic Ticket.Value
            Current == Seen || return Current
            CpuPause()
            # A worker that spins without ever reaching a safepoint blocks
            # garbage collection for the whole process.
            GC.safepoint()
        end

        lock(Barrier)
        try
            # Published before the ticket is re-read, and the coordinator reads
            # this flag only after storing the ticket. Both are sequentially
            # consistent, so at least one side sees the other and a worker can
            # never park on a ticket that has already been handed out.
            @atomic Sleeping.Value = 1
            while (@atomic Ticket.Value) == Seen
                wait(Barrier)
            end
            @atomic Sleeping.Value = 0
            return @atomic Ticket.Value
        finally
            unlock(Barrier)
        end
    end

    @inline function PostTicket!(Pool::WorkerPool, WorkerIndex::Int, Generation::Int)
        @atomic Pool.Tickets[WorkerIndex].Value = Generation
        if (@atomic Pool.Sleepers[WorkerIndex].Value) == 1
            Barrier = Pool.Barriers[WorkerIndex]
            lock(Barrier)
            try
                notify(Barrier)
            finally
                unlock(Barrier)
            end
        end
        return nothing
    end

    function WorkerLoop(Pool::WorkerPool, WorkerIndex::Int)
        Ticket = Pool.Tickets[WorkerIndex]
        Completed = Pool.Completed[WorkerIndex]
        Sleeping = Pool.Sleepers[WorkerIndex]
        Barrier = Pool.Barriers[WorkerIndex]
        Seen = 0
        while true
            Seen = AwaitTicket(Ticket, Sleeping, Barrier, Seen)
            try
                # The pool outlives the world age it was created in, so a body
                # whose method was defined later - any closure from a script or
                # the REPL - is only callable through invokelatest.
                Base.invokelatest(@atomic(Pool.Job), WorkerIndex, Pool.Count)
            catch Failure
                # Symbolicating a backtrace from several workers at once crashes
                # the runtime, so hand the raw frames to the coordinator.
                Pool.Failures[WorkerIndex] = (Failure, catch_backtrace())
            end
            @atomic Completed.Value = Seen
        end
        return nothing
    end

    function MakeWorkerPool()
        ThreadIds = Base.Threads.threadpooltids(:default)
        Count = min(length(ThreadIds), WorkerCount())
        Pool = WorkerPool(
            Task[], Count,
            [PaddedCounter() for _ in 1:Count],
            [PaddedCounter() for _ in 1:Count],
            [PaddedCounter() for _ in 1:Count],
            [Base.Threads.Condition() for _ in 1:Count],
            Atomic{Int}(0),
            Any[nothing for _ in 1:Count],
            nothing, 0, false,
        )
        # Worker 1 is whichever task opens the region, so only 2:Count are tasks.
        for WorkerIndex in 2:Count
            push!(Pool.Tasks, SpawnPinnedWorker(ThreadIds[WorkerIndex]) do
                WorkerLoop(Pool, WorkerIndex)
            end)
        end
        return Pool
    end

    const PoolSlot = Ref{Union{Nothing, WorkerPool}}(nothing)
    const PoolLock = ReentrantLock()

    function ActivePool()
        WorkerCount() > 1 || return nothing
        Pool = PoolSlot[]
        Pool === nothing || return Pool
        return lock(PoolLock) do
            if PoolSlot[] === nothing
                PoolSlot[] = MakeWorkerPool()
            end
            return PoolSlot[]
        end
    end

    # A body that opened a second region would reset the shared cursor underneath
    # the outer loop, so one region runs at a time and anything nested falls back
    # to running serially inside its caller.
    @inline function EnterRegion!(Pool::WorkerPool)
        _, Entered = @atomicreplace Pool.Active false => true
        return Entered
    end

    @inline LeaveRegion!(Pool::WorkerPool) = (@atomic Pool.Active = false; nothing)

    function AwaitCompletion(Completed::PaddedCounter, Generation::Int)
        Spins = 0
        while (@atomic Completed.Value) != Generation
            CpuPause()
            GC.safepoint()
            Spins += 1
            if Spins >= CoordinatorSpinBudget
                yield()
                Spins = 0
            end
        end
        return nothing
    end

    function RunRegion!(Body::F, Pool::WorkerPool) where {F}
        fill!(Pool.Failures, nothing)
        @atomic Pool.Job = Body
        Generation = @atomic Pool.Generation += 1
        for WorkerIndex in 2:Pool.Count
            PostTicket!(Pool, WorkerIndex, Generation)
        end

        try
            Body(1, Pool.Count)
        catch Failure
            Pool.Failures[1] = (Failure, catch_backtrace())
        end

        # Joining every worker before returning is what keeps the caller's
        # Bumper buffers valid for the whole region.
        for WorkerIndex in 2:Pool.Count
            AwaitCompletion(Pool.Completed[WorkerIndex], Generation)
        end

        for Failure in Pool.Failures
            Failure === nothing || throw(CapturedException(Failure[1], Failure[2]))
        end
        return nothing
    end

    """
        ForEachWorker!(Body)

    Run `Body(WorkerIndex, WorkerCount)` once per worker and return only after
    every one of them has finished. Falls back to a single serial call when the
    pool is unavailable or a region is already open.
    """
    function ForEachWorker!(Body::F) where {F}
        Pool = ActivePool()
        if Pool !== nothing && EnterRegion!(Pool)
            try
                RunRegion!(Body, Pool)
            finally
                LeaveRegion!(Pool)
            end
            return nothing
        end
        Body(1, 1)
        return nothing
    end

    """
        ForEachIndex!(Body, Indices, BatchSize)

    Apply `Body` to every index of `Indices` exactly once, handing out
    contiguous batches of `BatchSize` through a shared cursor. Spatially sorted
    particles give contiguous partitions very different amounts of work, so
    workers take small batches rather than a fixed slice each.
    """
    function ForEachIndex!(Body::F, Indices::AbstractUnitRange, BatchSize::Int) where {F}
        isempty(Indices) && return nothing
        Pool = ActivePool()
        if Pool !== nothing && length(Indices) > BatchSize && EnterRegion!(Pool)
            try
                Cursor = Pool.Cursor
                Last = last(Indices)
                Cursor[] = first(Indices)
                RunRegion!(Pool) do WorkerIndex, Count
                    BatchStart = atomic_add!(Cursor, BatchSize)
                    while BatchStart <= Last
                        for Index in BatchStart:min(BatchStart + BatchSize - 1, Last)
                            Body(Index)
                        end
                        BatchStart = atomic_add!(Cursor, BatchSize)
                    end
                end
            finally
                LeaveRegion!(Pool)
            end
            return nothing
        end

        for Index in Indices
            Body(Index)
        end
        return nothing
    end

end
