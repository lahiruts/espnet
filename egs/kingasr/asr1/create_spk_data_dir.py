import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()


with open(args.input_dir+"/data.json", encoding='utf-8') as f:
    data_json = json.load(f)
    

#group by spk
utts_dict = data_json['utts']
spk2recs = {}
for utt_id, rec in utts_dict.items():
    spk_id = rec['utt2spk']
    rec['utt_id'] = utt_id
    spk2recs.setdefault(spk_id, []).append(rec)
    

all_spk_ids = sorted(spk2recs.keys())

text = []
utt2spk = []
spk_dir = args.output_dir+"/"
os.system("mkdir -p "+args.output_dir)
for idx, spk in enumerate(all_spk_ids):
    recs = spk2recs[spk]
    # spk_dir = args.output_dir+"/%d"%(idx+1)
    # os.system("mkdir -p "+spk_dir)
    # text = []
    # utt2spk = []
    for rec in recs:
        rec_text = rec['output'][0]['rec_text'].replace("<eos>","").strip()
        if len(rec_text) == 0:
            print("skip", rec)
            continue
        text.append("%s %s\n"%(rec['utt_id'], rec_text))
        utt2spk.append("%s %s\n"%(rec['utt_id'], spk))
    text = sorted(text)
    utt2spk = sorted(utt2spk)
with open(spk_dir+"/text",'w', encoding='utf-8') as f:
    f.writelines(text)
with open(spk_dir+"/utt2spk",'w', encoding='utf-8') as f:
    f.writelines(utt2spk)
    
    
    
    
    
    
    

