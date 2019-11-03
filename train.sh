MAX_EPOCHS=25
# EPOCH=1
SIZE_EPOCH=80000

while [[ $EPOCH -le $MAX_EPOCHS ]]; do
    echo "$EPOCH"
    echo 'start training'
    if [[ $EPOCH -eq 1 ]]; then
        let STEP=EPOCH*SIZE_EPOCH
        python trainAttentionBox.py 0 attentionBox-final --init_weights resnet-50-model.caffemodel --step $SIZE_EPOCH
    else
        let STEP_OLD=(EPOCH - 1)*SIZE_EPOCH
        let STEP=EPOCH*SIZE_EPOCH
        python trainAttentionBox.py 0 attentionBox-final --restore attentionBox-final_iter_$STEP_OLD.solverstate --step $SIZE_EPOCH
    fi
    echo 'training done'
    echo 'start validation'
    python testAttentionBox.py 0 attentionBox-final --init_weights attentionBox-final_iter_$STEP.caffemodel --dataset subVal2014_2
    echo 'validation done'
    echo 'start evaluation'
    echo "$EPOCH" >>trainEval.txt
    python evalCOCO.py attentionBox-final --dataset subVal2014_2 --useSegm False >>trainEval.txt
    echo 'evaluation done'
    let EPOCH=EPOCH+1
done
