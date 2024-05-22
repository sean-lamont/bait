# Install requirements
conda create -n 'bait' python=3.10 ipython numpy
conda install torch (correct version)
pip install tqdm loguru deepspeed "lightning[extra]" transformers tensorboard openai rank_bm25 lean-dojo==1.6 wandb pyfarmhash pymongo torch_geometric plotly igraph einops dill peft graphviz
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Prepare data and download models

- set up wandb api key
- (optional) download cached lean traces 
- run data/LeanDojo/process_data.sh
- download retriever
- run experiments.models.end_to_end.tactic_models.retrieval.index, with above checkpoint, corpus in LeanDojo/data/benchmark/corpus.jsonl
- Can now run experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/leandojo to run novel premises 
