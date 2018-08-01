#!/bin/bash
# Copyright 2018 Fano Labs (Lahiru Samarakoon)

dir=data_before
for x in data_universal/train data_D/test data_D/train_cv10 data_D/train_tr90_sp \
         data_D_C1/train data_D_C2/train data_D_C1_C2/train; do
  echo `wc -l $dir/$x/text`
  mv $dir/$x/text $dir/$x/text.orig
  python3 local/convert_to_tranditional_chinese.py $dir/$x/text.orig $dir/$x/text

done
exit 0
