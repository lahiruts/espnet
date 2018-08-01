#!/bin/bash

# Copyright 2018 Fano Labs (Lahiru Samarakoon)


# general configuration
#backend=chainer
backend=pytorch
stage=4     # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
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
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=1


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
#datype=bias_embedding_input
datype=fgate
#daconfig=7   #input the configs 3_1_2_5 format
daconfig=7_1000_-1   #input the configs 3_1_2_5 format
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


pos=before
#pos=after
recog_set=test
rtask=test
train_set=train
train_dev=data_${pos}/data_D/train_cv10

data_D=data_${pos}/data_D
data_C1=data_${pos}/data_C1
data_C2=data_${pos}/data_C2
data_EN=data_${pos}/data_EN

dump_D=dump_D
dump_C1=dump_C1
dump_C2=dump_C2
dump_EN=dump_EN

data_universal=data_${pos}/data_universal #Mainly used for Universal Dictionary
data_D_C1=data_${pos}/data_D_C1 
data_D_C2=data_${pos}/data_D_C2 
data_D_C1_C2=data_${pos}/data_D_C1_C2 


feat_dt_dir=$train_dev

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 0: Combine training data directories"
    mkdir -p $data_universal/train
    mkdir -p $data_D_C1/train
    mkdir -p $data_D_C2/train
    mkdir -p $data_D_C1_C2/train

    #Creating the universal training set.  Validation set will be from the domain.	
    #utils/combine_data.sh  $data_universal/train $data_D/train_tr90_sp  $data_C1/train_tr90_nodup $data_C2/train_tr90_nodup $data_EN/train_trim
    cat $dump_D/train_tr90_sp/deltafalse/feats.scp $dump_C1/train_tr90_nodup/deltafalse/feats.scp $dump_C2/train_tr90_nodup/deltafalse/feats.scp $dump_EN/train_trim/deltafalse/feats.scp | sort > $data_universal/train/feats.scp   

 
    echo `wc -l $data_D/train_tr90_sp/text`
    echo `wc -l $data_C1/train_tr90_nodup/text`
    echo `wc -l $data_C2/train_tr90_nodup/text`
    echo `wc -l $data_EN/train_trim/text`
    echo `wc -l $data_universal/train/text`
    utils/fix_data_dir.sh $data_universal/train
    echo `wc -l $data_universal/train/text`
    echo `wc -l $data_universal/train/feats.scp`

    #Combining D and C1
    utils/combine_data.sh  $data_D_C1/train $data_D/train_tr90_sp  $data_C1/train_tr90_nodup
    cat $dump_D/train_tr90_sp/deltafalse/feats.scp $dump_C1/train_tr90_nodup/deltafalse/feats.scp | sort > $data_D_C1/train/feats.scp
    utils/fix_data_dir.sh $data_D_C1/train

    #Combining D and C2
    utils/combine_data.sh  $data_D_C2/train $data_D/train_tr90_sp  $data_C2/train_tr90_nodup
    cat $dump_D/train_tr90_sp/deltafalse/feats.scp $dump_C2/train_tr90_nodup/deltafalse/feats.scp | sort > $data_D_C2/train/feats.scp
    utils/fix_data_dir.sh $data_D_C2/train 

    #Combining D, C1, and C2
    utils/combine_data.sh  $data_D_C1_C2/train $data_D/train_tr90_sp  $data_C1/train_tr90_nodup $data_C2/train_tr90_nodup
    cat $dump_D/train_tr90_sp/deltafalse/feats.scp $dump_C1/train_tr90_nodup/deltafalse/feats.scp $dump_C2/train_tr90_nodup/deltafalse/feats.scp | sort > $data_D_C1_C2/train/feats.scp
    utils/fix_data_dir.sh $data_D_C1_C2/train
fi


dict=$data_universal/lang_1char/train_units.txt
echo "dictionary: ${dict}"
nlsyms=$data_universal/lang_1char/non_lang_syms.txt
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 1: Universal Dictionary preparation"
    mkdir -p $data_universal/lang_1char/

    echo "make a non-linguistic symbol list"
    printf "<SPK/>\n<NON/>\n<D>\n<C1>\n<C2>\n<EN>" > ${nlsyms}
    #cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} $data_universal/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

fi


if [ ${stage} -le 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 3: Json Data Preparation"

    #In domain validation set
    data2json.sh --feat $train_dev/feats.scp --nlsyms ${nlsyms} \
         $train_dev ${dict} > ${train_dev}/data.json
    #Universal training set
    #data2json.sh --feat $data_universal/${train_set}/feats.scp --nlsyms ${nlsyms} \
    #     $data_universal/${train_set} ${dict} > $data_universal/${train_set}/data.json
   

    #D, C1
    #data2json.sh --feat $data_D_C1/${train_set}/feats.scp --nlsyms ${nlsyms} \
    #     $data_D_C1/${train_set} ${dict} > $data_D_C1/${train_set}/data.json
    
    #D, C2
    #data2json.sh --feat $data_D_C2/${train_set}/feats.scp --nlsyms ${nlsyms} \
    #     $data_D_C2/${train_set} ${dict} > $data_D_C2/${train_set}/data.json 

    #D, C1, C2
    data2json.sh --feat $data_D_C1_C2/${train_set}/feats.scp --nlsyms ${nlsyms} \
         $data_D_C1_C2/${train_set} ${dict} > $data_D_C1_C2/${train_set}/data.json

   
   data2json.sh --feat $data_D/${rtask}/feats.scp --nlsyms ${nlsyms} \
         $data_D/${rtask} ${dict} > $data_D/${rtask}/data.json


fi


dataset=data_D_C1_C2
feat_tr_dir=data_${pos}/$dataset/train

if [ -z ${tag} ]; then
    expdir=exp/${ngpu}_${backend}_${pos}_${da}_${datype}_${daconfig}_${dataset}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
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
        --da ${da} \
        --datype ${datype} \
        --daconfig ${daconfig} \
        --epochs ${epochs}
fi



if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
	
	feat_recog_dir=$data_D/${rtask}
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
	    --datype ${datype} \
            --daconfig ${daconfig} \
	    --da ${da} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} &
        wait
        #    --rnnlm ${lmexpdir}/rnnlm.model.best \
        #    --lm-weight ${lm_weight} &
        #wait

        score_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi


exit 0
