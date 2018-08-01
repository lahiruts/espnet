#!/bin/bash

# Copyright 2018 Fano Labs (Lahiru Samarakoon)


# general configuration
#backend=chainer
backend=pytorch
stage=6     # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=2          # number of gpus ("0" uses cpu, otherwise use gpu)
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
epochs=20


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


recog_set=test
rtask=test
train_set=train
train_dev=data_D/train_cv10

data_D=data_D
data_C1=data_C1
data_C2=data_C2
data_EN=data_EN

dump_D=dump_D
dump_C1=dump_C1
dump_C2=dump_C2
dump_EN=dump_EN

data_universal=data_universal #Mainly used for Universal Dictionary
data_D_C1=data_D_C1 
data_D_C2=data_D_C2 
data_D_C1_C2=data_D_C1_C2 


feat_dt_dir=$train_dev

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 0: Combine training data directories"
    mkdir -p $data_universal/train
    mkdir -p $data_D_C1/train
    mkdir -p $data_D_C2/train
    mkdir -p $data_D_C1_C2/train

    #Creating the universal training set.  Validation set will be from the domain.	
    utils/combine_data.sh  $data_universal/train $data_D/train_tr90_sp  $data_C1/train_tr90_nodup $data_C2/train_tr90_nodup $data_EN/train_trim
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
    echo "stage 0: Universal Dictionary preparation"
    mkdir -p $data_universal/lang_1char/

    echo "make a non-linguistic symbol list"
    printf "<SPK/>\n<NON/>" > ${nlsyms}
    #cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} $data_universal/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

fi


if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Json Data Preparation"

    #In domain validation set
    data2json.sh --feat $train_dev/feats.scp --nlsyms ${nlsyms} \
         $train_dev ${dict} > ${train_dev}/data.json
    #Universal training set
    data2json.sh --feat $data_universal/${train_set}/feats.scp --nlsyms ${nlsyms} \
         $data_universal/${train_set} ${dict} > $data_universal/${train_set}/data.json
   

    #D, C1
    data2json.sh --feat $data_D_C1/${train_set}/feats.scp --nlsyms ${nlsyms} \
         $data_D_C1/${train_set} ${dict} > $data_D_C1/${train_set}/data.json
    
    #D, C2
    data2json.sh --feat $data_D_C2/${train_set}/feats.scp --nlsyms ${nlsyms} \
         $data_D_C2/${train_set} ${dict} > $data_D_C2/${train_set}/data.json 

    #D, C1, C2
    data2json.sh --feat $data_D_C1_C2/${train_set}/feats.scp --nlsyms ${nlsyms} \
         $data_D_C1_C2/${train_set} ${dict} > $data_D_C1_C2/${train_set}/data.json


fi


dataset=data_D_C1_C2
feat_tr_dir=$dataset/train

if [ -z ${tag} ]; then
    expdir=exp/${ngpu}_${backend}_${dataset}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
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
        --epochs ${epochs}
fi


dumpdir=dump_D
feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
if [ ${stage} -le 5 ]; then
	mkdir -p ${feat_recog_dir}
	dump.sh --cmd "$train_cmd" --nj 5 --do_delta $do_delta \
            ${data_D}/${rtask}/feats.scp ${data_D}/cmvn.ark exp/dump_feats/${rtask} ${feat_recog_dir}

	data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} ${data_D}/${rtask} ${dict} > ${feat_recog_dir}/data.json

fi


if [ ${stage} -le 6 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

        # split data
        #data=$data_D/${rtask}
        #split_data.sh --per-utt ${data} ${nj};
        #sdata=${data}/split${nj}utt;


        # feature extraction
        #feats="ark,s,cs:apply-cmvn --norm-vars=true ${data_D}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        #if ${do_delta}; then
        #feats="$feats add-deltas ark:- ark:- |"
        #fi

        # make json labels for recognition
        #data2json.sh --feat ${data}/feats.scp --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json
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
# general configuration
#backend=chainer
backend=pytorch
stage=2     # start from 0 if you need to start from data preparation
gpu=1            # will be deprecated, please use ngpu
ngpu=2          # number of gpus ("0" uses cpu, otherwise use gpu)
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
epochs=20
#eps=0.000000000204082 #The value for 7 GPUs when using adadelta


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

# data
CANTO_HK=/media/data/speech_data/Cantonese-HK/

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

data=data_CANTO-HK
audio_dir=$data/all

train_set=train_tr90_sp
train_dev=train_cv10
recog_set=test

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    #local/prepare_cantonese_data.sh ${CANTO_HK} ${audio_dir}
    # remove space in text
    cp ${audio_dir}/text ${audio_dir}/text.org
    paste -d " " <(cut -f 1 -d" " ${audio_dir}/text.org) <(cut -f 2- -d" " ${audio_dir}/text.org | tr -d " ") \
    > ${audio_dir}/text
    #rm ${audio_dir}/text.org
    
    #for x in all; do
    #    sed -i.bak -e "s/$/ | sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
    #done
    #create train, test and dev sets
    utils/subset_data_dir_tr_cv.sh $audio_dir $data/train $data/test
    utils/subset_data_dir_tr_cv.sh $data/train $data/train_tr90 $data/train_cv10

    #local/cantones_format_data.sh
fi


#feat_tr_di=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
#feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 5 data/test exp/make_fbank/test ${fbankdir}
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 4 data/train_cv10 exp/make_fbank/train_cv10 ${fbankdir}

    # make a dev set
    #utils/subset_data_dir.sh --first data/train 4000 data/${train_dev}
    #n=$[`cat data/train/segments | wc -l` - 4000]
    #utils/subset_data_dir.sh --last data/train $n data/train_nodev

    # make a training set
    #utils/data/remove_dup_utts.sh 300 data/train_tr90 data/train_tr90_nodup

    # speed-perturbed
    utils/perturb_data_dir_speed.sh 0.9 data/train_tr90 data/temp1
    utils/perturb_data_dir_speed.sh 1.0 data/train_tr90 data/temp2
    utils/perturb_data_dir_speed.sh 1.1 data/train_tr90 data/temp3
    utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
    #rm -r data/temp1 data/temp2 data/temp3
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 41 data/${train_set} exp/make_fbank/${train_set} ${fbankdir}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    split_dir=`echo $PWD | awk -F "/" '{print $NF "/" $(NF-1)}'`
    dump.sh --cmd "$train_cmd" --nj 41 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
nlsyms=data/lang_1char/non_lang_syms.txt
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    printf "<SPK/>\n<NON/>" > ${nlsyms}
    #cut -f 2- data/${train_set}/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
fi


exit 0
# you can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs${batchsize_lm}
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} data/train_tr90_nodup/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --gpu ${gpu} \
        --verbose 1 \
        --batchsize ${batchsize_lm} \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${ngpu}_${backend}_${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    #expdir=exp/${backend}_${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_ctc${ctctype}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
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
        --train-feat scp:${feat_tr_dir}/feats.scp \
        --valid-feat scp:${feat_dt_dir}/feats.scp \
        --train-label ${feat_tr_dir}/data.json \
        --valid-label ${feat_dt_dir}/data.json \
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
        --epochs ${epochs}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=5

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}

        # split data
        data=data/${rtask}
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true data/${train_set}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
        feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        data2json.sh --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        gpu=-1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
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

