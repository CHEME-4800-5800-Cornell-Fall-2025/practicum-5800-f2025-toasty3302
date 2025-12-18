"""
    MyClassicalHopfieldNetworkModel

A mutable struct representing a Classical Hopfield Network.

### Fields
- `W::Array{Float32,2}`: The weight matrix of the network (N × N symmetric matrix with zero diagonal)
- `b::Array{Float32,1}`: The bias vector for each neuron (typically zeros for classical Hopfield)
- `energy::Dict{Int64,Float32}`: Dictionary mapping memory index to its energy value
"""
mutable struct MyClassicalHopfieldNetworkModel

    # data -
    W::Array{Float32,2}         # weight matrix (N × N)
    b::Array{Float32,1}         # bias vector (N × 1)
    energy::Dict{Int64,Float32} # energy for each stored memory

    # constructor -
    MyClassicalHopfieldNetworkModel() = new()
end