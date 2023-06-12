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

@testset "add/remove_snapshot" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    g = rand_graph(10, 20)
    tsg2 = add_snapshot(tsg, 3, g)
    @test tsg2.num_nodes == [10 for i in 1:6]
    @test tsg2.num_edges == [20 for i in 1:6]
    @test tsg2.snapshots[3] == g
    @test tsg2.num_snapshots == 6
    @test tsg.num_nodes == [10 for i in 1:5]
    @test tsg.num_edges == [20 for i in 1:5]
    @test tsg.snapshots[2] === tsg2.snapshots[2]
    @test tsg.snapshots[3] === tsg2.snapshots[4]
    @test length(tsg.snapshots) == 5
    @test tsg.num_snapshots == 5
    
    
    tsg3 = remove_snapshot(tsg, 3)
    @test tsg3.num_nodes == [10 for i in 1:4]
    @test tsg3.num_edges == [20 for i in 1:4]
    @test tsg3.snapshots == snapshots[[1,2,4,5]]
end


# @testset "add/remove_snapshot!" begin
#     snapshots = [rand_graph(10, 20) for i in 1:5]
#     tsg = TemporalSnapshotsGNNGraph(snapshots)
#     g = rand_graph(10, 20)
#     tsg2 = add_snapshot!(tsg, 3, g)
#     @test tsg2.num_nodes == [10 for i in 1:6]
#     @test tsg2.num_edges == [20 for i in 1:6]
#     @test tsg2.snapshots[3] == g
#     @test tsg2.num_snapshots == 6
#     @test tsg2 === tsg
    
#     tsg3 = remove_snapshot!(tsg, 3)
#     @test tsg3.num_nodes == [10 for i in 1:4]
#     @test tsg3.num_edges == [20 for i in 1:4]
#     @test length(tsg3.snapshots) === 4
#     @test tsg3 === tsg
# end

@testset "show" begin
    snapshots = [rand_graph(10, 20) for i in 1:5]
    tsg = TemporalSnapshotsGNNGraph(snapshots)
    @test sprint(show,tsg) == "TemporalSnapshotsGNNGraph(5) with no data"
    @test sprint(show, MIME("text/plain"), tsg; context=:compact => true) == "TemporalSnapshotsGNNGraph(5) with no data"
    @test sprint(show, MIME("text/plain"), tsg; context=:compact =>  false) == "TemporalSnapshotsGNNGraph:\n  num_nodes: [10, 10, 10, 10, 10]\n  num_edges: [20, 20, 20, 20, 20]\n  num_snapshots: 5"
    tsg.tgdata.x=rand(4)
    @test sprint(show,tsg) == "TemporalSnapshotsGNNGraph(5) with x: 4-element data"
end

#     @test sprint(show, MIME("text/plain"), rand_graph(10, 20); context=:compact => true) == "GNNGraph(10, 20) with no data"