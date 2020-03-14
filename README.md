# data2text-transformer
Code for **Enhanced Transformer Model for Data-to-Text Generation** [\[PDF\]](https://www.aclweb.org/anthology/D19-5615/) (Gong, Crego, Senellart; WNGT2019).
Much of this code is adapted from an earlier fork of [XLM](https://github.com/facebookresearch/XLM).

**EMNLP-WNGT2019 [Evaluation Results](https://docs.google.com/spreadsheets/d/18ZYbK67uJ2yGlJ48IRWEIkVHN_fP135Ecg-BVPhJeXI/edit#gid=2090491847)** (SYSTRAN-AI & SYSTRAN-AI-detok)

## Dataset and Preprocessing

The boxscore-data json files can be downloaded from the [boxscore-data repo](https://github.com/harvardnlp/boxscore-data).

Assuming the RotoWire json files reside at `./rotowire`, the following commands will preprocess the data

### Step1: Data extraction 

```
python scripts/data_extract.py -d rotowire/train.json -o rotowire/train
```

In this step, we:

* Convert the tables into a sequence of records: `train.gtable`
* Extract the summary and transform entity tokens (e.g., **Kobe Bryant** -> **Kobe_Bryant**): `train.summary`
* Mark the occurrances of records in the summary: `train.gtable_label` and `train.summary_label`

### Step2: Extract vocabulary

```
python scripts/extract_vocab.py -t rotowire/train.gtable -s rotowire/train.summary
```
It will generate vocabulary files for each of them:

* `rotowire/train.gtable_vocab`
* `rotowire/train.summary_vocab`

### Step3: Binarize the data

```
python model/preprocess_summary_data.py --summary rotowire/train.summary \
                                        --summary_vocab rotowire/train.summary_vocab \
                                        --summary_label rotowire/train.summary_label
                                        
python model/preprocess_table_data.py --table rotowire/train.gtable \
                                      --table_label rotowire/train.gtable_label \
                                      --table_vocab rotowire/train.gtable_vocab
```
And we finally get the training data:
* Input record sequences: `train.gtable.pth`
* Output summaries: `train.summary.pth`

## Model Training
```
MODELPATH=$PWD/model
export PYTHONPATH=$MODELPATH:$PYTHONPATH

python $MODELPATH/train.py

## main parameters
--model_path "experiments"
--exp_name "baseline"
--exp_id "try1"

## data location / training objective
--train_cs_table_path rotowire/train.gtable.pth        # record data for content selection (CS) training
--train_sm_table_path rotowire/train.gtable.pth        # record data for data2text summarization (SM) training
--train_sm_summary_path rotowire/train.summary.pth     # summary data for data2text summarization (SM) training
--valid_table_path rotowire/valid.gtable.pth           # input record data for validation
--valid_summary_path rotowire/valid.summary.pth        # output summary data for validation
--cs_step True                                         # enable content selection training objective
--lambda_cs "1"                                        # CS training coefficient
--sm_step True                                         # enable summarization objective
--lambda_sm "1"                                        # SM training coefficient
    
## transformer parameters
--label_smoothing 0.05                                 # label smoothing
--share_inout_emb True                                 # share the embedding and softmax weights in decoder
--emb_dim 512                                          # embedding size
--enc_n_layers 1                                       # number of encoder layers
--dec_n_layers 6                                       # number of decoder layers
--dropout 0.1                                          # dropout

## optimization
--save_periodic 1                                      # save model every N epoches
--batch_size 6                                         # batch size (number of examples)
--beam_size 4                                          # beam search in generation
--epoch_size 1000                                      # number of examples per epoch
--eval_bleu True                                       # evaluate the BLEU score
--validation_metrics valid_mt_bleu                     # validation metrics
```

## Generation

Use the following commands to generate from the above models:

Download the baseline model from: [link](https://drive.google.com/open?id=1o4kx0xJPbYser2RmpTHa-3aDlBl_M_uu)

```
MODEL_PATH=experiments/baseline/try1/best-valid_mt_bleu.pth
INPUT_TABLE=rotowire/valid.gtable
OUTPUT_SUMMARY=rotowire/valid.gtable_out

python model/summarize.py 
    --model_path $MODEL_PATH
    --table_path $INPUT_TABLE
    --output_path $OUTPUT_SUMMARY
    --beam_size 4
```

### Postprocessing after generation
In the preprocessing step1 (data extraction), the entity tokens are transformed (e.g., **Kobe Bryant** -> **Kobe_Bryant**). Here we revert such transformation:

```
cat ${OUTPUT_SUMMARY} | sed 's/_/ /g' > ${OUTPUT_SUMMARY}_txt
```

## Evaluation

### Content-oriented evaluation

We use the code in [https://github.com/ratishsp/data2text-1](https://github.com/ratishsp/data2text-1) for evaluation.

Metrics of RG, CS, CO are computed using the below commands.

#### Prepare dataset for the IE system
```
~/anaconda2/bin/python data_utils.py 
    -mode make_ie_data                      # mode
    -input_path "../rotowire"               # rotowire data path
    -output_fi "roto-ie.h5"                 # output filename
```
#### Generate h5 file for output summary
```
~/anaconda2/bin/python data_utils.py 
    -mode prep_gen_data                     # mode 
    -gen_fi ${OUTPUT_SUMMARY}_txt           # generated summary (postprocessed) 
    -dict_pfx "roto-ie"                     # dict prefix of IE system
    -output_fi ${OUTPUT_SUMMARY}_txt.h5     # output h5 filename
    -input_path ../rotowire                 # rotowire data path
```

#### Evaluate RG metrics
```
th extractor.lua 
    -gpuid 1 
    -datafile roto-ie.h5                    # dataset of IE system
    -preddata ${OUTPUT_SUMMARY}_txt.h5         # generated h5 file in the previous step
    -dict_pfx roto-ie                       # dict prefix of IE system
    -just_eval
```
#### Evaluate CS and CO metrics
```
~/anaconda2/bin/python non_rg_metrics.py roto-gold-val.h5-tuples.txt ${OUTPUT_SUMMARY}_txt.h5-tuples.txt
```

### BLEU evaluation

The BLEU evaluation script can be obtained from [Moses](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl):
```
perl multi-bleu.perl ${reference_summary} < ${generated_summary}
```
