DELETE_DIR=True
Q_results_dir=$AAAI_WORK_DIR/test_DP_results

# unit test of DP with small congestion trace
python $AAAI_ROOT_DIR/DP/PPC_DP_wrapper_test.py --delete_dir $DELETE_DIR --base_results_dir $Q_results_dir

python $AAAI_ROOT_DIR/DP/plot_DP_performance_only.py --base_results_dir $Q_results_dir
