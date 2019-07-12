## Graph Embedding Pytorch Implementation

Pytorch version of Graph Embeddings

* How to run: just run "python main.py"
    * See 'config.json' for detailed settings and options.
    * Training options are managed by 'config.json'.
    
* Model Implemented: GCN

* Data Implemented: Cora, PPI

* Current Obstacles:
    * Minor problems in handling the filestream Logger output path.
    * Not sufficient Models & Data implemented
    * Current Model's output is not suitable for PPI Data

* Remaining Jobs:
    * Implement More Model and Data
    * Fix logger to properly output the file
    * GCN's Last aggregation by Softmax is not Adequate to PPI data
