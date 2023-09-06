#leanpkg configure
#leanproject get-mathlib-cache
#bash ./_target/deps/mathlib/scripts/mk_all.sh
#leanpkg build  # needs to build everything, takes 20-30m

# skip tests from original repo

#mkdir ./data

#lean --run src/tools/all_decls.lean

#python3 python/dhash.py ./data/mathlib_decls.log ./data/

RAW_DATA_DIR="./raw_data"
N_WORKERS=5
REC_LIMIT=5000 # conservative value: 3000
DEPTH_LIMIT=100 # conservative value: 64
WEIGHT_LIMIT=2000 # conservative value: 1500
DECLS_PER_SHARD=100

python3 ./python/parallel_gen_data.py ./data/train_decls.log $RAW_DATA_DIR $N_WORKERS $REC_LIMIT $DEPTH_LIMIT $WEIGHT_LIMIT $DECLS_PER_SHARD


