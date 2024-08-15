# generate the intra-attention w/o causal results
# results: execution_plans/intra_SP4_fob=0/1
./scripts/intra_attn_gen.sh 2>&1 | tee ./results/search_intra_SP=4_noncausal_all.log
