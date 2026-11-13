#!/bin/bash
set -e
# Repeater script for: NPU lowering
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aie.dma_bd' op At most four data layout transformation dimensions may be provided.
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='a.mlir.prj/aiecc_failure_1794412700_985568.mlir'
PASS_PIPELINE='any(aie-materialize-runtime-sequences,aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
