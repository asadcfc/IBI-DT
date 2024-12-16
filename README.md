# IBI-DT
IBI-DT (Individualized Bayesian Inference - Decision Tree) algorithm advances the IBI methodology by integrating decision trees to examine complex traits at the subgroup level in cancer research. This approach offers a framework for exploring and understanding the interactions of genomic variants and their impacts on cancer.

# IBIDT_core
IBIDT_core provides the structure of 'n' decision trees. This included all the essential components required for an in-depth analysis of tree-based data. The key elements included are:


Marginal: Provides corresponding marginal.

Node: Represents the best node.

Level: Indicates the depth.

Branch: Details the diverging paths (direction).

OR_Details: Provides the odds ration and corresponding details.

Tree: Tree number

# SingleNode_and_PathwayPair_analysis

This module is an integral part of our framework, designed to utilize the tree information and provide key statistics.

α Statistic for Single Nodes: The file calculates the α statistic for single nodes and saves it in a file named SingleNode_analysis.csv. 

β Statistic for Pathway Pairs: In addition to single node analysis, this file also computes the β statistic for pathway pairs and saves it in PathwayPair_analysis.csv.
