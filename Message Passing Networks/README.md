
# Message Passing Neural Networks:

Message passing networks (MPN) aim to generalize convolution operation to irregular domains. To achieve this, they imitate the structure of graphs and 
perform message passing between the nodes of that graph.

First of all, a graph with optional nodes and edges is constructed, and each of these nodes and edges is represented by a feature vector, which is commonly
called "embeddings". Then, a series of information propagation (message passing) steps are carried out on this graph to make the nodes and edges interact 
with each other. In other words, the nodes share some kinds of information with each other by means of the edges, which is why it is called message passing
networks.  

Each message-passing step (an iteration) in MPN can be considered to be a layer in common neural networks, but note that there is actually no physical 
layer in our graph neural network for those iterations. In each iteration, all the nodes connect to their neighboring nodes and gather information. As 
the number of iterations increases, the nodes get a chance to go in deeper levels of the graph, and connect to the further nodes. The action "connecting"
mentioned at this point should not be misunderstood; new edges are not built, but actually the nodes receive some of kinds of information from their 
further counterparts, and this shared information is actually the embeddings of the nodes. In each iteration, these embeddings are updated.
