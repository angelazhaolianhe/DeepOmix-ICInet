
# Biological knowledge graph-guided investigation of immune therapy response in cancer with graph neural network

This project implements the ICInet (DeepOmix-ICI) framework, a deep learning model that leverages biological knowledge graphs to predict immune therapy response in cancer patients. This framework is based on the research paper "Biological knowledge graph-guided investigation of immune therapy response in cancer with graph neural network".

## Project Overview

The ICInet model aims to improve the prediction of immune checkpoint inhibitor (ICI) treatment response by integrating patient gene expression profiles with prior biological knowledge, specifically using graph neural networks (GNNs).

**Key Features:**
-   **Knowledge Graph Integration:** Utilizes protein-protein interaction (PPI) networks, gene ontology, and pathway databases to construct a comprehensive gene-gene knowledge graph.
-   **ICI Target-Proximal Gene Identification:** Employs the PageRank algorithm to identify genes functionally close to known ICI targets (e.g., PD1, PD-L1).
-   **Graph Neural Networks:** Uses GNNs to learn representations from gene expression data on the constructed biological networks.
-   **Patient-Level Prediction:** Predicts patient response (responder/non-responder) to immune therapy.
-   **Robust Evaluation:** Supports intra-cohort, cross-cohort, and k-fold cross-validation strategies as described in the original paper.


## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/angelazhaolianhe/DeepOmix-ICInet.git 
cd DeepOmix-ICInet
