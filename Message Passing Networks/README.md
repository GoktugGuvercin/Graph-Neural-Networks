
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

The information is propagated (spread) across the nodes and edges in each iteration. Each node talks to their neighbors, collect information from them, and update its own internal representation (embedding vector). Contextual information is spread more and more as the number of iteration increases. In that way, each node starts to learn wider parts of the graph, and its knowledge base covers both larger regions and relationships of the nodes in the graph. When, message passing process is completed, an output graph with updated "context-aware" node and edge feature vectors shows up.

# Message Passing Process:

In message passing process, the graph denoted by $G = (V, E)$ is at first constructed, where $V = \\{1, 2, \cdots, i, j, \cdots, n\\}$ is the set of node indices and $E = \\{(1, 2), \cdots, (i, j), \cdots \\}$ is a collection of node pairs for edge representation. Graph neural network architectures aim to learn the representation of nodes and encode their possible edge connections. Hence, an embedding feature vector is defined for each node and each edge between mutually connected any two nodes.

$h_{ij}^0 \rightarrow$ The embedding of edge $(i, j)$ in iteration $0$.

$h_{i}^0 \rightarrow$ The embedding of node $i$ in iteration $0$. 

Initial embedding vectors can be randomly or manually defined. It is also possible to use simple CNN or RNN to extract initial embedding features. In first iteration, the nodes do not know anything about the graph and even their neighbors. 
