#!/bin/bash
# Copyright 2018 Fano Labs (Lahiru Samarakoon)

for x in data_D/test data_D/train data_D/train_cv10 data_D/train_tr90 data_D/train_tr90_sp \
         data_D_C1/train data_D_C2/train data_D_C1_C2/train; do
  echo `wc -l $x/text`
  mv $x/text $x/text.orig
  python3 local/convert_to_tranditional_chinese.py $x/text.orig $x/text

done
exit 0
