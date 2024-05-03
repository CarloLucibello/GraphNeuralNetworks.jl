
GNNGraphs._rand_dense_vector(A::CUMAT_T) = CUDA.randn(size(A, 1))
