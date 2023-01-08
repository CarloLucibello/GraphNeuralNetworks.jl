@testset "Utils" begin @testset "edge encoding/decoding" begin
    # not is_bidirected
    n = 5
    s = [1, 1, 2, 3, 3, 4, 5]
    t = [1, 3, 1, 1, 2, 5, 5]

    # directed=true
    idx, maxid = GNNGraphs.edge_encoding(s, t, n)
    @test maxid == n^2
    @test idx == [1, 3, 6, 11, 12, 20, 25]

    sdec, tdec = GNNGraphs.edge_decoding(idx, n)
    @test sdec == s
    @test tdec == t

    n1, m1 = 10, 30
    g = rand_graph(n1, m1)
    s1, t1 = edge_index(g)
    idx, maxid = GNNGraphs.edge_encoding(s1, t1, n1)
    sdec, tdec = GNNGraphs.edge_decoding(idx, n1)
    @test sdec == s1
    @test tdec == t1

    # directed=false
    idx, maxid = GNNGraphs.edge_encoding(s, t, n, directed = false)
    @test maxid == n * (n + 1) ÷ 2
    @test idx == [1, 3, 2, 3, 7, 14, 15]

    mask = s .> t
    snew = copy(s)
    tnew = copy(t)
    snew[mask] .= t[mask]
    tnew[mask] .= s[mask]
    sdec, tdec = GNNGraphs.edge_decoding(idx, n, directed = false)
    @test sdec == snew
    @test tdec == tnew

    n1, m1 = 6, 8
    g = rand_graph(n1, m1)
    s1, t1 = edge_index(g)
    idx, maxid = GNNGraphs.edge_encoding(s1, t1, n1, directed = false)
    sdec, tdec = GNNGraphs.edge_decoding(idx, n1, directed = false)
    mask = s1 .> t1
    snew = copy(s1)
    tnew = copy(t1)
    snew[mask] .= t1[mask]
    tnew[mask] .= s1[mask]
    @test sdec == snew
    @test tdec == tnew
end end
