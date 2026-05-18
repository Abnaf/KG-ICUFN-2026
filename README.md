# KG-ICUFN-2026

G-Core-RCA-GNN: Unified Graph Neural Network Benchmark for 5G Core Failure Localization.

This repository contains a unified benchmarking framework for Root Cause Analysis (RCA) and failure localization within 5G Core networks. It bridges tabular telemetry data with Graph Neural Networks (GNNs) and Knowledge Graph (KG)-inspired architectures by dynamic graph construction via k-Nearest Neighbors ($k$-NN). The framework evaluates baseline models alongside a Proposed Attention-based Knowledge Graph Failure Localization Model over multiple stochastic initializations, reporting performance metrics with mean and standard deviation.

Architecture Overview
5G Core network telemetry yields highly complex, non-linear dependencies across network functions (AMF, SMF, UPF, etc.). This repository builds an explicit topological graph structure directly out of raw telemetry metrics to allow structural information routing via message passing.

[Raw 5G Telemetry CSV] 
       │
       ▼ (Pre-processing & Min-Max Scaling)
[Continuous Feature Matrix (X)] ───► [k-NN Graph Generation (k=10)] ───► [PyG Data Object Structure]
                                                                                  │
   ┌────────────────┬─────────────────┼────────────────┬─────────────────┐        │
   ▼                ▼                 ▼                ▼                 ▼        ▼
[GCN]           [KGroot]           [KAFD]           [LAMs]            [CF-GNN] ... [Proposed Model]
   │                │                 │                │                 │        │
   └────────────────┴─────────────────┼────────────────┴─────────────────┘        │
                                      
        
