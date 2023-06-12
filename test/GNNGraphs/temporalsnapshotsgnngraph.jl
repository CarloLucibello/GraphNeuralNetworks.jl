@testset "Constructor array TemporalSnapshotsGNNGraph" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg.num_nodes == [10 for i in 1:5]
    @test tsg.num_edges == [20 for i in 1:5]
    wrsnapshots = [rand_graph(10,20), rand_graph(12,22)]
    @test_throws AssertionError TemporalSnapshotsGNNGraph(wrsnapshots)
end

@testset "==" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg1 = TemporalSnapshotsGNNGraph(snapshots)
    tsg2 = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg1 == tsg2
    tsg3 = TemporalSnapshotsGNNGraph(snapshots[1:3])
    @test tsg1 != tsg3
    @test tsg1 !== tsg3
end

@testset "getindex" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg[3] == snapshots[3]
    @test tsg[[1,2]] == TemporalSnapshotsGNNGraph([10,10], [20,20], 2, snapshots[1:2], tsg.tgdata)
end

@testset "getproperty" begin
    x = rand(10)
    snapshots = [rand_graph(10, 20, ndata = x) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test tsg.tgdata == DataStore()
    @test tsg.x == tsg.ndata.x == [x for i in 1:5]
    @test_throws KeyError tsg.ndata.w
    @test_throws ArgumentError tsg.w
end

@testset "add/remove_snapshot" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    g = rand_graph(10, 20)
    tsg = add_snapshot(tsg, 3, g)
    @test tsg.num_nodes == [10 for i in 1:6]
    @test tsg.num_edges == [20 for i in 1:6]
    @test tsg.snapshots[3] == g
    tsg = remove_snapshot(tsg, 3)
    @test tsg.num_nodes == [10 for i in 1:5]
    @test tsg.num_edges == [20 for i in 1:5]
    @test tsg.snapshots == snapshots
end

@testset "show" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test sprint(show,tsg) == "TemporalSnapshotsGNNGraph(5) with no data"
    @test sprint(show, MIME("text/plain"), tsg; context=:compact => true) == "TemporalSnapshotsGNNGraph(5) with no data"
    @test sprint(show, MIME("text/plain"), tsg; context=:compact =>  false) == "TemporalSnapshotsGNNGraph:\n  num_nodes: [10, 10, 10, 10, 10]\n  num_edges: [20, 20, 20, 20, 20]\n  num_snapshots: 5"
    tsg.tgdata.x=rand(4)
    @test sprint(show,tsg) == "TemporalSnapshotsGNNGraph(5) with x: 4-element data"
end
