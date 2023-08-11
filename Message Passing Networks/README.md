
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

$h_{i}^n \rightarrow$ The embedding of node $i$ in iteration $n$. 

Initial embedding vectors can be randomly or manually defined. It is also possible to use simple CNN or RNN to extract initial embedding features. In first iteration, the nodes do not know anything about the graph and even their neighbors. As message passing operations are performed, the nodes get more familiar with their neighbors, but also learn what their neighbors know and see about local neighborhood. In that way, each node starts to learn local topology on the graph. If N number of message passing iterations proceed, each node would access and meet all the nodes located N step further.

Message passing operation is split into $2$ parts:
1. Message creation per node
2. Node update 

In message creation, the embedding vectors of target node, its one neighbor and the connection between them are aggregated and passed to a learnable function such as an MLP to generate a message. At this point, one of the approaches to aggregate these embedding vectors is concatenation. The output of MLP would be a message of just one neighbor, so same operation is repeated for all neighbors and output message per neighbor is combined in a permutation invariant way, which can be summation, max or mean. The following formula is general denotion of message creation process:

$m_v^{(k+1)} = \sum_{v \in N(u)} M(h_u^k, \\,\\, h_v^k, \\,\\, h_{u\\,v})$

In node update, the created message is combined with current embedding vector of target node and passed to another learnable function like MLP to generate next embedding vector of the node $u$:

$h_u^{k+1} = U(h_u^k, \\,\\, m_v^{(k+1)})$


