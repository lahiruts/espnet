#!/bin/bash

# general configuration
backend=pytorch
stage=2     # start from 0 if you need to start from data preparation
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
batchsize_lm=64
lm_weight=0.0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'


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


# data
working_dir=smartone_data
expdir=$working_dir/model
data=$working_dir/eval


#Prepare the Data for Kaldi format
#base_file=$(basename $file)
#data=${expdir}/${base_file}
#mkdir -p $data
#echo "$base_file $file" > $data/wav.scp
#echo "$base_file $base_file" > $data/spk2utt
#cp $data/spk2utt $data/utt2spk
#Create a default text file
#echo "$base_file 咁你今晚食咩做GYM啊" > $data/text

if [ ${stage} -le 1 ]; then
	#Extract features
	fbankdir=$data/fbank/
	steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 1 $data $data/make_fbank/ ${fbankdir}
fi

dict=data_universal/lang_1char/train_units.txt
echo "dictionary: ${dict}"
nlsyms=data_universal/lang_1char/non_lang_syms.txt

if [ ${stage} -le 2 ]; then
    echo "stage 2: Decoding"
    nj=1

        decode_dir=${working_dir}/decode

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

        #${decode_cmd} JOB=1:${nj} ${decode_dir}/log/decode.JOB.log \
        #    asr_recog.py \
        #    --gpu ${gpu} \
        #    --backend ${backend} \
        #    --recog-feat "$feats" \
        #    --recog-label ${data}/data.json \
        #    --result-label ${decode_dir}/data.JOB.json \
        #    --model ${working_dir}/model.${recog_model}  \
        #    --model-conf ${working_dir}/model.conf  \
        #    --beam-size ${beam_size} \
        #    --penalty ${penalty} \
        #    --maxlenratio ${maxlenratio} \
        #    --minlenratio ${minlenratio} \
        #    --ctc-weight ${ctc_weight} \
        #    --rnnlm ${lmexpdir}/rnnlm.model.best \
        #    --lm-weight ${lm_weight} &
        #wait

	
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


        #score_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
    echo "Finished"
fi

#result=`grep "prediction" ${decode_dir}/log/decode.1.log | awk '{print $NF}' | sed 's/<eos>//'`

echo "the result is $result"
end_time=$SECONDS
total_time=$((end_time-start_time))
echo "Total time: $total_time"
