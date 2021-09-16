# Message Passing

The message passing is initiated by [`propagate`](@ref)
and can be customized for a specific layer by overloading the methods
[`compute_message`](@ref), [`update_node`](@ref), and [`update_edge`](@ref).


The message passing corresponds to the following operations 

```math
\begin{aligned}
\mathbf{m}_{j\to i} &= \phi(\mathbf{x}_i, \mathbf{x}_j, \mathbf{e}_{j\to i}) \\
\mathbf{x}_{i}' &= \gamma_x(\mathbf{x}_{i}, \square_{j\in N(i)}  \mathbf{m}_{j\to i})\\
\mathbf{e}_{j\to i}^\prime &=  \gamma_e(\mathbf{e}_{j \to i},\mathbf{m}_{j \to i})
\end{aligned}
```
where ``\phi`` is expressed by the [`compute_message`](@ref) function, 
``\gamma_x`` and ``\gamma_e`` by [`update_node`](@ref) and [`update_edge`](@ref)
respectively.

See [`GraphConv`](ref) and [`GATConv`](ref)'s implementations as usage examples. 
