#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=vggblstm     # encoder architecture type
elayers=3
eunits=1024
eprojs=1024
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
encoder_dropout=0.0
decoder_dropout=0.2
# decoder related
dlayers=2
dunits=1024
# attention related
#atype=location
#atype=multi_location
atype=factorized_location
adim=1024
aconv_chans=10
aconv_filts=100
gatt_dim=256
att_scale=3.0
gatt_scale=0.7
#gatt_dim=0

#gunits=1024
gunits=0
gprojs=512

num_save_attention=10
# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=20
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
opt=adadelta
epochs=5
patience=0
eps=1e-3

# rnnlm related
lm_layers=2
lm_units=650
lm_opt=sgd        # or adam
lm_sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
lm_batchsize=64   # batch size in LM training
lm_epochs=20      # if the data size is large, we can reduce this
lm_patience=3
lm_maxlen=100     # if sentence length > lm_maxlen, lm_batchsize is automatically reduced
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
lm_weight=0.3
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.6
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0


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

train_set=train_tr90_sp
train_dev=train_cv10
recog_set=test

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt


if [ -z ${tag} ]; then
    expname=edropout_${encoder_dropout}_dec_dropout_${decoder_dropout}_patience_${patience}_gunits_${gunits}_gprojs_${gprojs}_${gatt_scale}_${att_scale}_${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
#expdir=exp/edropout_0.2_dec_dropout_0.0_patience_0_gunits_1024_gprojs_512_0.7_3.0_train_tr90_sp_pytorch_vggblstm_e3_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_multi_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs20_mli800_mlo150

expdir=exp/0.7_3.0_train_tr90_sp_pytorch_vggblstm_e3_subsample1_2_2_1_1_unit1024_proj1024_d2_unit1024_factorized_location_aconvc10_aconvf100_mtlalpha0.5_adadelta_sampprob0.0_bs20_mli800_mlo150

decode_dir=decode_test_beam20_emodel.acc.best_p0.0_len0.0-0.0_ctcw0.6_rnnlm0.3_2layer_unit650_sgd_bs64
num_test_speakers=5

lmexpdir=exp/train_rnnlm_pytorch_2layer_unit650_sgd_bs64
main_data_dir=$expdir/data
main_oracle_data=$expdir/data_oracle


if [ ${stage} -le 2 ]; then
        echo "stage 1: Process the firstpass decoding outputs"

        mkdir -p $main_data_dir
        grep "asr:489" ${expdir}/${decode_dir}/log/decode.* | awk '{print $7}' | sed 's/\[//' | sed 's/\]//' | sed 's/://' > $main_data_dir/temp_utts
        #grep "asr:507" ${expdir}/${decode_dir}/log/decode.* | awk '{print $7}' | sed 's/\[//' | sed 's/\]//' | sed 's/://' > $main_data_dir/temp_utts
        grep "prediction" ${expdir}/${decode_dir}/log/decode.* | awk '{print $NF}' | sed 's/<eos>//' > $main_data_dir/temp_predictions
        paste $main_data_dir/temp_utts $main_data_dir/temp_predictions | sort | uniq > $main_data_dir/outputs
        cp -r data/$recog_set/* $main_data_dir/
        rm -rf $main_data_dir/split*
        rm -rf $main_data_dir/log
        rm -rf $main_data_dir/dump_feats
        cp dump/test/deltafalse/feats.scp $main_data_dir/
        grep -v "*" $main_data_dir/outputs > $main_data_dir/text
     
        ./utils/fix_data_dir.sh $main_data_dir 
        ./utils/split_data.sh $main_data_dir $num_test_speakers
 
        mkdir -p $main_oracle_data
        cp -r data/$recog_set/*  $main_oracle_data/
        cp dump/test/deltafalse/feats.scp $main_oracle_data/
        ./utils/fix_data_dir.sh $main_oracle_data
        ./utils/split_data.sh $main_oracle_data $num_test_speakers

        for val in `seq 1 ${num_test_speakers}`; do 
              working_dir=$expdir/${val}
              echo $working_dir
    	      data_unsup=$working_dir/data
              data_oracle=$main_oracle_data/split${num_test_speakers}/${val}
              #mkdir -p $data_oracle
              mkdir -p $data_unsup
              cp -r $main_data_dir/split${num_test_speakers}/${val}/* $data_unsup/
              #cp -r $data_unsup/* $data_oracle/

              ./utils/fix_data_dir.sh $data_unsup
              #cp data/$recog_set/text $data_oracle/    # has the oracle transcripts for second pass decode scoring.
              ./utils/fix_data_dir.sh $data_oracle

 
              data2json.sh --feat $data_unsup/feats.scp --nlsyms ${nlsyms} $data_unsup ${dict} > $data_unsup/data.json
              data2json.sh --feat $data_oracle/feats.scp --nlsyms ${nlsyms} $data_oracle ${dict} > $data_oracle/data.json
	done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding after adaptation"
   
second_pass_decode=$expdir/decode_second_pass
mkdir -p $second_pass_decode

for val in `seq 1 ${num_test_speakers}`; do
   (
   
    data_oracle=$main_oracle_data/split${num_test_speakers}/${val} 
    export CUDA_VISIBLE_DEVICES=${val}    
    working_dir=$expdir/${val}

    ${cuda_cmd} --gpu ${ngpu} ${working_dir}/train.log \
        asr_adapt.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${working_dir}/results \
        --model ${expdir}/results/${recog_model}  \
        --tensorboard-dir ${working_dir}/tensorboard/${expdir} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${working_dir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${working_dir}/data/data.json \
        --valid-json ${working_dir}/data/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --gunits ${gunits} \
        --gprojs ${gprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --num-save-attention ${num_save_attention} \
        --gatt-dim ${gatt_dim} \
        --att-scale ${att_scale} \
        --gatt-scale ${gatt_scale} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --sampling-probability ${samp_prob} \
        --opt ${opt} \
        --dropout-rate ${encoder_dropout} \
        --dropout-rate-decoder ${decoder_dropout} \
        --sortagrad ${sortagrad} \
        --epochs ${epochs} \
        --eps ${eps} \
        --patience ${patience}

        nj=5
        decode_dir=decode_secondpass_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}

        # split data
        splitjson.py --parts ${nj} ${data_oracle}/data.json
 
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${working_dir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${data_oracle}/split${nj}utt/data.JOB.json \
            --result-label ${working_dir}/${decode_dir}/data.${val}.JOB.json \
            --model ${working_dir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight}

       #cp ${working_dir}/${decode_dir}/data.json $second_pass_decode/data.${val}.json

     ) &
     pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Second Pass Decoding Finished"

for val in `seq 1 ${num_test_speakers}`; do
  working_dir=$expdir/${val}
  decode_dir=decode_secondpass_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
  second_pass_decode=$expdir/decode_second_pass
  cp ${working_dir}/${decode_dir}/data.* $second_pass_decode/
done

score_sclite.sh --nlsyms ${nlsyms} $second_pass_decode ${dict}

fi

exit 0
