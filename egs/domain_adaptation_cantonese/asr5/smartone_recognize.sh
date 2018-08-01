#!/bin/bash

# general configuration
backend=pytorch
stage=3     # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=2          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot


# feature configuration
do_delta=false # true when using CNN

# rnnlm related

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'



# rnnlm related
batchsize_lm=16
lm_weight=0.3
unit=650

da=1
datype=fgate
daconfig=7_1000_-1
#input wav file

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

start_time=$SECONDS

pos=before
data_universal=data_${pos}/data_universal

# data
working_dir=smartone_data

expdir=$working_dir/baseline
mkdir -p $expdir

data=$working_dir/smartone_kaldi_tags

dict=$data_universal/lang_1char/train_units.txt
echo "dictionary: ${dict}"
nlsyms=$data_universal/lang_1char/non_lang_syms.txt


if [ ${stage} -le 0 ]; then
        #Extract features
        echo "stage 0: Feature Extraction"
        fbankdir=$data/fbank/
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 12 $data $data/make_fbank/ ${fbankdir}
fi


if [ ${stage} -le 1 ]; then
        #Dump
	mv smartone_data/smartone_kaldi/feats.scp smartone_data/smartone_kaldi/feats_orig.scp
        echo "stage 1: Dump"
	dump.sh --cmd "$train_cmd" --nj 12 --do_delta $do_delta \
        	smartone_data/smartone_kaldi/feats_orig.scp data_before/data_D/cmvn.ark smartone_data/smartone_kaldi/dump_feats/ smartone_data/smartone_kaldi 
fi


if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Json Data Preparation"

    #In domain validation set
   data2json.sh --feat smartone_data/smartone_kaldi_tags/feats.scp --nlsyms ${nlsyms} \
         smartone_data/smartone_kaldi_tags ${dict} > smartone_data/smartone_kaldi_tags/data.json

   data2json.sh --feat smartone_data/smartone_kaldi/feats.scp --nlsyms ${nlsyms} \
         smartone_data/smartone_kaldi ${dict} > smartone_data/smartone_kaldi/data.json

fi

feat_recog_dir=smartone_data/smartone_kaldi_tags
expdir=smartone_data/fgate/results
if [ ${stage} -le 3 ]; then
    echo "stage 3: Decoding"
    nj=32

        decode_dir=${expdir}/decode
	splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0


        ${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${decode_dir}/data.JOB.json \
	    --model smartone_data/fgate/model.acc.best  \
            --model-conf smartone_data/fgate/model.conf  \
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


exit 0

lmexpdir=$data/train_rnnlm_2layer_bs${batchsize_lm}_unit${unit}_unify
mkdir -p ${lmexpdir}
if [ ${stage} -le 0 ]; then
    echo "stage 0: LM Preparation"
    lmdatadir=$data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 -l ${nlsyms} $data/unify_text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt
    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l ${nlsyms} $data/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --gpu 2 \
	--backend ${backend} \
        --verbose 1 \
        --batchsize ${batchsize_lm} \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
	--unit ${unit} \
        --dict ${dict}
fi


if [ ${stage} -le 1 ]; then
	#Extract features
        echo "stage 1: Feature Extraction"
	fbankdir=$data/fbank/
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 12 $data $data/make_fbank/ ${fbankdir}
fi


if [ ${stage} -le 2 ]; then
    echo "stage 2: Decoding"
    nj=12

        decode_dir=${data}/decode_${lm_weight}_${unit}_unify

        # split data
        split_data.sh --per-utt ${data} ${nj};
        sdata=${data}/split${nj}utt;

        # feature extraction
        feats="ark,s,cs:apply-cmvn --norm-vars=true ${data}/cmvn.ark scp:${sdata}/JOB/feats.scp ark:- |"
        if ${do_delta}; then
        feats="$feats add-deltas ark:- ark:- |"
        fi

        # make json labels for recognition
        data2json.sh --nlsyms ${nlsyms} ${data} ${dict} > ${data}/data.json

        #### use CPU for decoding
        ngpu=0

	
	${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-feat "$feats" \
            --recog-label ${data}/data.json \
            --result-label ${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
 	    --rnnlm ${lmexpdir}/rnnlm.model.best \
	    --lm-unit ${unit} \
	    --lm-weight ${lm_weight} &
  	wait


        score_sclite.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}
    echo "Finished"
fi

#result=`grep "prediction" ${decode_dir}/log/decode.1.log | awk '{print $NF}' | sed 's/<eos>//'`

#echo "the result is $result"
end_time=$SECONDS
total_time=$((end_time-start_time))
echo "Total time: $total_time"
