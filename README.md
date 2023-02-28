# Sparse and Local Hypergraph Reasoning Networks

Reasoning about the relationships between entities from input facts (e.g., whether Ari is a grandparent of Charlie) generally requires explicit consideration of other entities that are not mentioned in the query (e.g., the parents of Charlie). In this paper, we present an approach for learning to solve problems of this kind in large, real-world domains, using sparse and local hypergraph neural networks (SpaLoc). SpaLoc is motivated by two observations from traditional logic-based reasoning: relational inferences usually apply locally (i.e., involve only a small number of individuals), and relations are usually sparse (i.e., only hold for a small percentage of tuples in a domain). We exploit these properties to make learning and inference efficient in very large domains by (1) using a sparse tensor representation for hypergraph neural networks, (2) applying a sparsification loss during training to encourage sparse representations, and (3) subsampling based on a novel information sufficiency–based sampling process during training. SpaLoc achieves state-of-the-art performance on several real-world, large-scale knowledge graph reasoning benchmarks, and is the first framework for applying hypergraph neural networks on real-world knowledge graphs with more than 10k nodes.