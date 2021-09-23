# Deprecated in v0.1 

@deprecate GINConv(nn; eps=0, aggr=+)  GINConv(nn, eps; aggr)

# TO Deprecate
# x, _ = propagate(l, g, l.aggr, x, e)

# # TODO deprecate
# propagate(l, g::GNNGraph, aggr, x, e=nothing) = propagate(l, g, aggr; x, e)

