"""
    rand_graph(n, m; bidirected=true, seed=-1, edge_weight = nothing, kws...)

Generate a random (Erdós-Renyi) `GNNGraph` with `n` nodes and `m` edges.

If `bidirected=true` the reverse edge of each edge will be present.
If `bidirected=false` instead, `m` unrelated edges are generated.
In any case, the output graph will contain no self-loops or multi-edges.

A vector can be passed  as `edge_weight`. Its length has to be equal to `m`
in the directed case, and `m÷2` in the bidirected one.

Use a `seed > 0` for reproducibility.

Additional keyword arguments will be passed to the [`GNNGraph`](@ref) constructor.

# Examples

```juliarepl
julia> g = rand_graph(5, 4, bidirected=false)
GNNGraph:
    num_nodes = 5
    num_edges = 4

julia> edge_index(g)
([1, 3, 3, 4], [5, 4, 5, 2])

# In the bidirected case, edge data will be duplicated on the reverse edges if needed.
julia> g = rand_graph(5, 4, edata=rand(16, 2))
GNNGraph:
    num_nodes = 5
    num_edges = 4
    edata:
        e => (16, 4)

# Each edge has a reverse
julia> edge_index(g)
([1, 3, 3, 4], [3, 4, 1, 3])

```
"""
function rand_graph(n::Integer, m::Integer; bidirected = true, seed = -1, edge_weight = nothing, kws...)
    if bidirected
        @assert iseven(m) "Need even number of edges for bidirected graphs, given m=$m."
    end
    m2 = bidirected ? m ÷ 2 : m
    return GNNGraph(Graphs.erdos_renyi(n, m2; is_directed = !bidirected, seed); edge_weight, kws...)
end

"""
    rand_heterograph(n, m; seed=-1, bidirected=false, kws...)

Construct an [`GNNHeteroGraph`](@ref) with number of nodes and edges 
specified by `n` and `m` respectively. `n` and `m` can be any iterable of pairs
specifing node/edge types and their numbers.

Use a `seed > 0` for reproducibility.

Setting `bidirected=true` will generate a bidirected graph, i.e. each edge will have a reverse edge.
Therefore, for each edge type `(:A, :rel, :B)` a corresponding reverse edge type `(:B, :rel, :A)`
will be generated.

Additional keyword arguments will be passed to the [`GNNHeteroGraph`](@ref) constructor.

# Examples

```julia-repl
julia> g = rand_heterograph((:user => 10, :movie => 20),
                            (:user, :rate, :movie) => 30)
GNNHeteroGraph:
  num_nodes: (:user => 10, :movie => 20)         
  num_edges: ((:user, :rate, :movie) => 30,)
```
"""
function rand_heterograph end

# for generic iterators of pairs
rand_heterograph(n, m; kws...) = rand_heterograph(Dict(n), Dict(m); kws...)

function rand_heterograph(n::NDict, m::EDict; bidirected = false, seed = -1, kws...)
    rng = seed > 0 ? MersenneTwister(seed) : Random.GLOBAL_RNG
    if bidirected
        return _rand_bidirected_heterograph(rng, n, m; kws...)
    end
    graphs = Dict(k => _rand_edges(rng, (n[k[1]], n[k[3]]), m[k]) for k in keys(m))
    return GNNHeteroGraph(graphs; num_nodes = n, kws...)
end

function _rand_bidirected_heterograph(rng, n::NDict, m::EDict; kws...)
    for k in keys(m)
        if reverse(k) ∈ keys(m)
            @assert m[k] == m[reverse(k)] "Number of edges must be the same in reverse edge types for bidirected graphs."
        else
            m[reverse(k)] = m[k]
        end
    end
    graphs = Dict{EType, Tuple{Vector{Int}, Vector{Int}, Nothing}}()
    for k in keys(m)
        reverse(k) ∈ keys(graphs) && continue
        s, t, val =  _rand_edges(rng, (n[k[1]], n[k[3]]), m[k])
        graphs[k] = s, t, val
        graphs[reverse(k)] = t, s, val
    end
    return GNNHeteroGraph(graphs; num_nodes = n, kws...)
end

function _rand_edges(rng, (n1, n2), m)
    idx = StatsBase.sample(rng, 1:(n1 * n2), m, replace = false)
    s, t = edge_decoding(idx, n1, n2)
    val = nothing
    return s, t, val
end

"""
    rand_bipartite_heterograph(n1, n2, m; [bidirected, seed, node_t, edge_t, kws...])
    rand_bipartite_heterograph((n1, n2), m; ...)
    rand_bipartite_heterograph((n1, n2), (m1, m2); ...)

Construct an [`GNNHeteroGraph`](@ref) with number of nodes and edges
specified by `n1`, `n2` and `m1` and `m2` respectively.

See [`rand_heterograph`](@ref) for a more general version.

# Keyword arguments

- `bidirected`: whether to generate a bidirected graph. Default is `true`.
- `seed`: random seed. Default is `-1` (no seed).
- `node_t`: node types. If `bipartite=true`, this should be a tuple of two node types, otherwise it should be a single node type.
- `edge_t`: edge types. If `bipartite=true`, this should be a tuple of two edge types, otherwise it should be a single edge type.
"""
function rand_bipartite_heterograph end

rand_bipartite_heterograph(n1::Int, n2::Int, m::Int; kws...) = rand_bipartite_heterograph((n1, n2), (m, m); kws...)

rand_bipartite_heterograph((n1, n2)::NTuple{2,Int}, m::Int; kws...) = rand_bipartite_heterograph((n1, n2), (m, m); kws...)

function rand_bipartite_heterograph((n1, n2)::NTuple{2,Int}, (m1, m2)::NTuple{2,Int}; bidirected=true, 
                        node_t = (:A, :B), edge_t = :to, kws...)
    if edge_t isa Symbol
        edge_t = (edge_t, edge_t)
    end
    return rand_heterograph(Dict(node_t[1] => n1, node_t[2] => n2), 
                            Dict((node_t[1], edge_t[1], node_t[2]) => m1, (node_t[2], edge_t[2], node_t[1]) => m2); 
                            bidirected, kws...)
end

"""
    knn_graph(points::AbstractMatrix, 
              k::Int; 
              graph_indicator = nothing,
              self_loops = false, 
              dir = :in, 
              kws...)

Create a `k`-nearest neighbor graph where each node is linked 
to its `k` closest `points`.  

# Arguments

- `points`: A num_features × num_nodes matrix storing the Euclidean positions of the nodes.
- `k`: The number of neighbors considered in the kNN algorithm.
- `graph_indicator`: Either nothing or a vector containing the graph assignment of each node, 
                     in which case the returned graph will be a batch of graphs. 
- `self_loops`: If `true`, consider the node itself among its `k` nearest neighbors, in which
                case the graph will contain self-loops. 
- `dir`: The direction of the edges. If `dir=:in` edges go from the `k` 
         neighbors to the central node. If `dir=:out` we have the opposite
         direction.
- `kws`: Further keyword arguments will be passed to the [`GNNGraph`](@ref) constructor.

# Examples

```juliarepl
julia> n, k = 10, 3;

julia> x = rand(3, n);

julia> g = knn_graph(x, k)
GNNGraph:
    num_nodes = 10
    num_edges = 30

julia> graph_indicator = [1,1,1,1,1,2,2,2,2,2];

julia> g = knn_graph(x, k; graph_indicator)
GNNGraph:
    num_nodes = 10
    num_edges = 30
    num_graphs = 2

```
"""
function knn_graph(points::AbstractMatrix, k::Int;
                   graph_indicator = nothing,
                   self_loops = false,
                   dir = :in,
                   kws...)
    if graph_indicator !== nothing
        d, n = size(points)
        @assert graph_indicator isa AbstractVector{<:Integer}
        @assert length(graph_indicator) == n
        # All graphs in the batch must have at least k nodes. 
        cm = StatsBase.countmap(graph_indicator)
        @assert all(values(cm) .>= k)

        # Make sure that the distance between points in different graphs
        # is always larger than any distance within the same graph.
        points = points .- minimum(points)
        points = points ./ maximum(points)
        dummy_feature = 2d .* reshape(graph_indicator, 1, n)
        points = vcat(points, dummy_feature)
    end

    kdtree = NearestNeighbors.KDTree(points)
    if !self_loops
        k += 1
    end
    sortres = false
    idxs, dists = NearestNeighbors.knn(kdtree, points, k, sortres)

    g = GNNGraph(idxs; dir, graph_indicator, kws...)
    if !self_loops
        g = remove_self_loops(g)
    end
    return g
end

"""
    radius_graph(points::AbstractMatrix, 
                 r::AbstractFloat; 
                 graph_indicator = nothing,
                 self_loops = false, 
                 dir = :in, 
                 kws...)

Create a graph where each node is linked 
to its neighbors within a given distance `r`.  

# Arguments

- `points`: A num_features × num_nodes matrix storing the Euclidean positions of the nodes.
- `r`: The radius.
- `graph_indicator`: Either nothing or a vector containing the graph assignment of each node, 
                     in which case the returned graph will be a batch of graphs. 
- `self_loops`: If `true`, consider the node itself among its neighbors, in which
                case the graph will contain self-loops. 
- `dir`: The direction of the edges. If `dir=:in` edges go from the
         neighbors to the central node. If `dir=:out` we have the opposite
         direction.
- `kws`: Further keyword arguments will be passed to the [`GNNGraph`](@ref) constructor.

# Examples

```juliarepl
julia> n, r = 10, 0.75;

julia> x = rand(3, n);

julia> g = radius_graph(x, r)
GNNGraph:
    num_nodes = 10
    num_edges = 46

julia> graph_indicator = [1,1,1,1,1,2,2,2,2,2];

julia> g = radius_graph(x, r; graph_indicator)
GNNGraph:
    num_nodes = 10
    num_edges = 20
    num_graphs = 2

```
"""
function radius_graph(points::AbstractMatrix, r::AbstractFloat;
                      graph_indicator = nothing,
                      self_loops = false,
                      dir = :in,
                      kws...)
    if graph_indicator !== nothing
        d, n = size(points)
        @assert graph_indicator isa AbstractVector{<:Integer}
        @assert length(graph_indicator) == n

        # Make sure that the distance between points in different graphs
        # is always larger than r.
        dummy_feature = 2r .* reshape(graph_indicator, 1, n)
        points = vcat(points, dummy_feature)
    end

    balltree = NearestNeighbors.BallTree(points)

    sortres = false
    idxs = NearestNeighbors.inrange(balltree, points, r, sortres)

    g = GNNGraph(idxs; dir, graph_indicator, kws...)
    if !self_loops
        g = remove_self_loops(g)
    end
    return g
end

function temporal_rand_radius_graph(number_nodes::Int, 
                      number_snapshots::Int,
                      speed::AbstractFloat,
                      r::AbstractFloat;
                      self_loops = false,
                      dir = :in,
                      kws...)
    points=rand(2, number_nodes)
    tg = Vector{GNNGraph}(undef, number_snapshots)
    for t in 1:number_snapshots
        tg[t] = radius_graph(points, r; graph_indicator = nothing, self_loops, dir, kws...)
        for i in 1:number_nodes
            ρ=2*speed*rand()-speed
            theta=2*pi*rand()
            points[1,i]=abs(points[1,i]+ρ*cos(theta))%1
            points[2,i]=abs(points[2,i]+ρ*sin(theta))%1
        end
    end
    return TemporalSnapshotsGNNGraph(tg)
end
