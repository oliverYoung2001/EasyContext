# Generate the intra-attention w/o causal results
# results: execution_plans/intra_SP4_fob=0/1
./scripts/intra_attn_gen.sh 2>&1 | tee ./results/search_intra_SP=4_noncausal_all.log

# Generate the intra-attention w causal results
# results: execution_plans/intra_SP4_fob=0/1_causal=True
./scripts/search_engine_qy.sh 2>&1 | tee ./results/search_intra_SP=4_causal_all.log
# use generate_intra_execution_plans

# Generate profile file: wrapper_intra_SP=8_all.log
./scripts/wrapper_qy.sh 2>&1 | tee ./prof_data/wrapper_intra_SP=8_all.log
# use mode='profile'

# Generate the inter-attention w causal results
# results: inter_SP8_fob=0/1
./scripts/search_engine_qy.sh 2>&1 | tee ./results/search_inter_SP=4_causal_all.log
# generate_inter_execution_plans

# Generate the inter-attention w/o causal results   # No !!!
# None

# Run the inter-attention w causal results
./scripts/wrapper_qy.sh 2>&1 | tee ./results_exp/wrapper_inter_SP=4,8_causal=True.log
# use mode='test'

# Run the inter-attention w/o causal results
./scripts/wrapper_qy.sh 2>&1 | tee ./results_exp/wrapper_inter_SP=4,8_causal=False.log
# use mode='test'

# End to End: HACK


# Ablation 1: Workload Allocation. Searching results vs expert-designed results


# Ablation 2: Non-fused vs min(Non-fused, Fused)
# parse "wrapper_intra_SP=8_all.log", Nh=1, noncausal, SP=(1,8), Sg=..., Fob=0/1, max(non-fused) vs max(all)

# Ablation 3: ILP vs Flexflow
# parse "SP=1,4_Sg=1k_causal_ablation01.log" or "SP=1,4_Sg=1k_causal_ablation1.log"

# Searching of Computation Workload Allocation Engine:
./scripts/search_engine_qy.sh 2>&1 | tee ./results_exp/search_engine_qy_N=?_locality_fob=?.log

srun -p arch -N 4 --ntasks-per-node=8 --gres=gpu:8 --mem 256G -K  -c 16 hostname