"""
    rand_graph([rng,] n, m; bidirected=true, edge_weight = nothing, kws...)

Generate a random (Erdós-Renyi) `GNNGraph` with `n` nodes and `m` edges.

If `bidirected=true` the reverse edge of each edge will be present.
If `bidirected=false` instead, `m` unrelated edges are generated.
In any case, the output graph will contain no self-loops or multi-edges.

A vector can be passed  as `edge_weight`. Its length has to be equal to `m`
in the directed case, and `m÷2` in the bidirected one.

Pass a random number generator as the first argument to make the generation reproducible.

Additional keyword arguments will be passed to the [`GNNGraph`](@ref) constructor.

# Examples

```jldoctest
julia> g = rand_graph(5, 4, bidirected=false)
GNNGraph:
    num_nodes = 5
    num_edges = 4

julia> edge_index(g)
([1, 3, 3, 4], [5, 4, 5, 2])

# In the bidirected case, edge data will be duplicated on the reverse edges if needed.
julia> g = rand_graph(5, 4, edata=rand(Float32, 16, 2))
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
function rand_graph(n::Integer, m::Integer; seed=-1, kws...)
    if seed != -1
        Base.depwarn("Keyword argument `seed` is deprecated, pass an rng as first argument instead.", :rand_graph)
        rng = MersenneTwister(seed)
    else
        rng = Random.default_rng()
    end
    return rand_graph(rng, n, m; kws...)
end

function rand_graph(rng::AbstractRNG, n::Integer, m::Integer; 
            bidirected::Bool = true, 
            edge_weight::Union{AbstractVector, Nothing} = nothing, kws...)
    if bidirected
        @assert iseven(m) lazy"Need even number of edges for bidirected graphs, given m=$m."
        s, t, _ = _rand_edges(rng, n, m ÷ 2; directed=false, self_loops=false)
        s, t = vcat(s, t), vcat(t, s)
        if edge_weight !== nothing
            edge_weight = vcat(edge_weight, edge_weight)
        end
    else
        s, t, _ = _rand_edges(rng, n, m; directed=true, self_loops=false)
    end
    return GNNGraph((s, t, edge_weight); num_nodes=n, kws...)
end

"""
    rand_heterograph([rng,] n, m; bidirected=false, kws...)

Construct an [`GNNHeteroGraph`](@ref) with random edges and with number of nodes and edges 
specified by `n` and `m` respectively. `n` and `m` can be any iterable of pairs
specifing node/edge types and their numbers.

Pass a random number generator as a first argument to make the generation reproducible.

Setting `bidirected=true` will generate a bidirected graph, i.e. each edge will have a reverse edge.
Therefore, for each edge type `(:A, :rel, :B)` a corresponding reverse edge type `(:B, :rel, :A)`
will be generated.

Additional keyword arguments will be passed to the [`GNNHeteroGraph`](@ref) constructor.

# Examples

```jldoctest
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
rand_heterograph(rng::AbstractRNG, n, m; kws...) = rand_heterograph(rng, Dict(n), Dict(m); kws...)

function  rand_heterograph(n::NDict, m::EDict; seed=-1, kws...)
    if seed != -1
        Base.depwarn("Keyword argument `seed` is deprecated, pass an rng as first argument instead.", :rand_heterograph)
        rng = MersenneTwister(seed)
    else
        rng = Random.default_rng()
    end
    return rand_heterograph(rng, n, m; kws...)
end

function rand_heterograph(rng::AbstractRNG, n::NDict, m::EDict; bidirected::Bool = false, kws...)
    if bidirected
        return _rand_bidirected_heterograph(rng, n, m; kws...)
    end
    graphs = Dict(k => _rand_edges(rng, (n[k[1]], n[k[3]]), m[k]) for k in keys(m))
    return GNNHeteroGraph(graphs; num_nodes = n, kws...)
end

function _rand_bidirected_heterograph(rng::AbstractRNG, n::NDict, m::EDict; kws...)
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


"""
    rand_bipartite_heterograph([rng,] 
                               (n1, n2), (m12, m21); 
                               bidirected = true, 
                               node_t = (:A, :B), 
                               edge_t = :to, 
                               kws...)

Construct an [`GNNHeteroGraph`](@ref) with random edges representing a bipartite graph.
The graph will have two types of nodes, and edges will only connect nodes of different types.

The first argument is a tuple `(n1, n2)` specifying the number of nodes of each type.
The second argument is a tuple `(m12, m21)` specifying the number of edges connecting nodes of type `1` to nodes of type `2` 
and vice versa.

The type of nodes and edges can be specified with the `node_t` and `edge_t` keyword arguments,
which default to `(:A, :B)` and `:to` respectively.

If `bidirected=true` (default), the reverse edge of each edge will be present. In this case
`m12 == m21` is required.

A random number generator can be passed as the first argument to make the generation reproducible.

Additional keyword arguments will be passed to the [`GNNHeteroGraph`](@ref) constructor.

See [`rand_heterograph`](@ref) for a more general version.

# Examples

```julia-repl
julia> g = rand_bipartite_heterograph((10, 15), 20)
GNNHeteroGraph:
  num_nodes: (:A => 10, :B => 15)
  num_edges: ((:A, :to, :B) => 20, (:B, :to, :A) => 20)

julia> g = rand_bipartite_heterograph((10, 15), (20, 0), node_t=(:user, :item), edge_t=:-, bidirected=false)
GNNHeteroGraph:
  num_nodes: Dict(:item => 15, :user => 10)
  num_edges: Dict((:item, :-, :user) => 0, (:user, :-, :item) => 20)
```
"""
rand_bipartite_heterograph(n, m; kws...) = rand_bipartite_heterograph(Random.default_rng(), n, m; kws...)

function rand_bipartite_heterograph(rng::AbstractRNG, (n1, n2)::NTuple{2,Int}, m; bidirected=true, 
                        node_t = (:A, :B), edge_t::Symbol = :to, kws...)
    if m isa Integer
        m12 = m21 = m
    else
        m12, m21 = m
    end

    return rand_heterograph(rng, Dict(node_t[1] => n1, node_t[2] => n2), 
                            Dict((node_t[1], edge_t, node_t[2]) => m12, (node_t[2], edge_t, node_t[1]) => m21); 
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

```jldoctest
julia> n, k = 10, 3;

julia> x = rand(Float32, 3, n);

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

```jldoctest
julia> n, r = 10, 0.75;

julia> x = rand(Float32, 3, n);

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
# References
Section B paragraphs 1 and 2 of the paper [Dynamic Hidden-Variable Network Models](https://arxiv.org/pdf/2101.00414.pdf)
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

"""
    rand_temporal_radius_graph(number_nodes::Int, 
                               number_snapshots::Int,
                               speed::AbstractFloat,
                               r::AbstractFloat;
                               self_loops = false,
                               dir = :in,
                               kws...)

Create a random temporal graph given `number_nodes` nodes and `number_snapshots` snapshots.
First, the positions of the nodes are randomly generated in the unit square. Two nodes are connected if their distance is less than a given radius `r`.
Each following snapshot is obtained by applying the same construction to new positions obtained as follows.
For each snapshot, the new positions of the points are determined by applying random independent displacement vectors to the previous positions. The direction of the displacement is chosen uniformly at random and its length is chosen uniformly in `[0, speed]`. Then the connections are recomputed.
If a point happens to move outside the boundary, its position is updated as if it had bounced off the boundary.

# Arguments

- `number_nodes`: The number of nodes of each snapshot.
- `number_snapshots`: The number of snapshots.
- `speed`: The speed to update the nodes.
- `r`: The radius of connection.
- `self_loops`: If `true`, consider the node itself among its neighbors, in which
                case the graph will contain self-loops. 
- `dir`: The direction of the edges. If `dir=:in` edges go from the
         neighbors to the central node. If `dir=:out` we have the opposite
         direction.
- `kws`: Further keyword arguments will be passed to the [`GNNGraph`](@ref) constructor of each snapshot.

# Example

```jldoctest
julia> n, snaps, s, r = 10, 5, 0.1, 1.5;

julia> tg = rand_temporal_radius_graph(n,snaps,s,r) # complete graph at each snapshot
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [90, 90, 90, 90, 90]
  num_snapshots: 5
```  

"""
function rand_temporal_radius_graph(number_nodes::Int, 
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
            ρ = 2 * speed * rand() - speed
            theta=2*pi*rand()
            points[1,i]=1-abs(1-(abs(points[1,i]+ρ*cos(theta))))
            points[2,i]=1-abs(1-(abs(points[2,i]+ρ*sin(theta))))
        end
    end
    return TemporalSnapshotsGNNGraph(tg)
end


function _hyperbolic_distance(nodeA::Array{Float64, 1},nodeB::Array{Float64, 1}; ζ::Real)
    if nodeA != nodeB
        a = cosh(ζ * nodeA[1]) * cosh(ζ * nodeB[1])
        b = sinh(ζ * nodeA[1]) * sinh(ζ * nodeB[1])
        c = cos(pi - abs(pi - abs(nodeA[2] - nodeB[2])))
        d = acosh(a - (b * c)) / ζ
    else
        d = 0.0
    end
    return d
end

"""
    rand_temporal_hyperbolic_graph(number_nodes::Int, 
                                   number_snapshots::Int;
                                   α::Real,
                                   R::Real,
                                   speed::Real,
                                   ζ::Real=1,
                                   self_loop = false,
                                   kws...)

Create a random temporal graph given `number_nodes` nodes and `number_snapshots` snapshots.
First, the positions of the nodes are generated with a quasi-uniform distribution (depending on the parameter `α`) in hyperbolic space within a disk of radius `R`. Two nodes are connected if their hyperbolic distance is less than `R`. Each following snapshot is created in order to keep the same initial distribution.

# Arguments

- `number_nodes`: The number of nodes of each snapshot.
- `number_snapshots`: The number of snapshots.
- `α`: The parameter that controls the position of the points. If `α=ζ`, the points are uniformly distributed on the disk of radius `R`. If `α>ζ`, the points are more concentrated in the center of the disk. If `α<ζ`, the points are more concentrated at the boundary of the disk.
- `R`: The radius of the disk and of connection.
- `speed`: The speed to update the nodes.
- `ζ`: The parameter that controls the curvature of the disk.
- `self_loops`: If `true`, consider the node itself among its neighbors, in which
                case the graph will contain self-loops.
- `kws`: Further keyword arguments will be passed to the [`GNNGraph`](@ref) constructor of each snapshot.

# Example

```jldoctest
julia> n, snaps, α, R, speed, ζ = 10, 5, 1.0, 4.0, 0.1, 1.0;

julia> thg = rand_temporal_hyperbolic_graph(n, snaps; α, R, speed, ζ)
TemporalSnapshotsGNNGraph:
  num_nodes: [10, 10, 10, 10, 10]
  num_edges: [44, 46, 48, 42, 38]
  num_snapshots: 5
```

# References
Section D of the paper [Dynamic Hidden-Variable Network Models](https://arxiv.org/pdf/2101.00414.pdf) and the paper 
[Hyperbolic Geometry of Complex Networks](https://arxiv.org/pdf/1006.5169.pdf)
"""
function rand_temporal_hyperbolic_graph(number_nodes::Int,
                                        number_snapshots::Int;
                                        α::Real,
                                        R::Real,
                                        speed::Real,
                                        ζ::Real=1,
                                        self_loop = false,
                                        kws...)
        @assert number_snapshots > 1 "The number of snapshots must be greater than 1"
        @assert α > 0 "α must be greater than 0"

        probabilities = rand(number_nodes)

        points = Array{Float64}(undef,2,number_nodes)
        points[1,:].= (1/α) * acosh.(1 .+ (cosh(α * R) - 1) * probabilities)
        points[2,:].= 2 * pi * rand(number_nodes)

        tg = Vector{GNNGraph}(undef, number_snapshots)

        for time in 1:number_snapshots
            adj = zeros(number_nodes,number_nodes)
            for i in 1:number_nodes
                for j in 1:number_nodes
                    if !self_loop && i==j
                        continue
                    elseif _hyperbolic_distance(points[:,i],points[:,j]; ζ) <= R
                        adj[i,j] = adj[j,i] = 1
                    end
                end
            end
            tg[time] = GNNGraph(adj)
            
            probabilities .= probabilities .+ (2 * speed * rand(number_nodes) .- speed)
            probabilities[probabilities.>1] .=  1 .- (probabilities[probabilities .> 1] .% 1)
            probabilities[probabilities.<0] .= abs.(probabilities[probabilities .< 0])

            points[1,:].= (1/α) * acosh.(1 .+ (cosh(α * R) - 1) * probabilities)
            points[2,:].= points[2,:] .+ (2 * speed * rand(number_nodes) .- speed)
        end
    return TemporalSnapshotsGNNGraph(tg)
end
