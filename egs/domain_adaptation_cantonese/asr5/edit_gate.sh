#!/bin/bash

# Copyright 2018 Fano Labs (Lahiru Samarakoon)

# general configuration
#backend=chainer
backend=pytorch
stage=0     # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# loss related
#ctctype=chainer
ctctype=warpctc
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=5
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=5
eps=1e-8

# rnnlm related
batchsize_lm=64
lm_weight=0.0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

#domain adaptation
da=1
#datype=bias_embedding
datype=fgate
#datype=fhl
daconfig=7_1000_-1   #input the configs 3_1_2_5 format
#daconfig=7   #input the configs 3_1_2_5 format
#daconfig=3_2   #input the configs 3_1_2_5 format
#daconfig=7_1000_-1   #input the configs 3_1_2_5 format
# data

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_universal=data_before/data_universal
dict=$data_universal/lang_1char/train_units.txt
echo "dictionary: ${dict}"
nlsyms=$data_universal/lang_1char/non_lang_syms.txt

for epochs in {1..8}; do

expdir=smartone_data/fgate
feat_tr_dir=smartone_data/data_fgate
workingdir=smartone_data/fgate/results_${epochs}
mkdir -p ${expdir}
mkdir -p ${workingdir}

if [ ${stage} -le 0 ]; then
    echo "stage 0: Network Editing"
    ${cuda_cmd} ${workingdir}/edit.log \
        asr_edit.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${workingdir} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${workingdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
	--model ${expdir}/model.${recog_model}  \
        --model-conf ${expdir}/model.conf  \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_tr_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --ctc_type ${ctctype} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --eps ${eps} \
        --da ${da} \
        --datype ${datype} \
        --daconfig ${daconfig} \
        --epochs ${epochs}
fi

feat_recog_dir=smartone_data/smartone_kaldi_tags
#workingdir=smartone_data/gate/results_${epochs}
if [ ${stage} -le 1 ]; then
    echo "stage 3: Decoding"
    nj=32

        decode_dir=${workingdir}/decode
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${decode_dir}/data.JOB.json \
            --model ${workingdir}/model.acc.best  \
            --model-conf ${expdir}/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --datype ${datype} \
            --daconfig ${daconfig} \
            --da ${da} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} &
        wait

        score_sclite.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}
    echo "Finished"
fi
done
exit 0
