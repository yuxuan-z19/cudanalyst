SUITE_ROOT="benchmark/hpc/npb"
WORKLOAD="CG"
WORKLOAD_DIR=$SUITE_ROOT/src/$WORKLOAD

python eval/openevolve/openevolve-run.py \
    $WORKLOAD_DIR/sol.init.cu \
    $SUITE_ROOT/eval.py \
    -c eval/config/openevolve.yml \
    -o out/npb-$WORKLOAD \
    -p $WORKLOAD_DIR \
    -a config/cugedit_debug.yml \
