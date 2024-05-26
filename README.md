# DeepLearningProject_2024

## Links to datasets used

*English Dataset*: [English Dataset](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)

Credits with respect to the English dataset and our thanks for making it publicly accessible go to:

B. Vidgen, T. Thrush, Z. Waseem, and D. Kiela,
“Learning from the Worst: Dynamically Generated
Datasets to Improve Online Hate Detection,” in Pro-
ceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th Interna-
tional Joint Conference on Natural Language Process-
ing (Volume 1: Long Papers) (C. Zong, F. Xia, W. Li,
and R. Navigli, eds.), (Online), pp. 1667–1682, Asso-
ciation for Computational Linguistics, Aug. 2021.

*Arabic Dataset*: [Arabic Dataset](https://sites.google.com/view/arabichate2022)

Credits with respect to the Arabic dataset and our thanks for making it publicly accessible go to:

H. Mubarak, S. Hassan, and S. A. Chowdhury, “Emojis
as Anchors to Detect Arabic Offensive Language and
Hate Speech,” Jan. 2022.

## Preprocessing of Arabic dataset

To preprocess the Arabic data for usage in the scripts below the reader is asked to consult the file *translate.ipynb*. Directions on how to proceed and set up [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate?tab=readme-ov-file) can be found in this file.

## The training/evaluation caller interface

The script *call_interface.py* can be used to reproduce training and a subsequent evaluation run of each of the models we've used. Alternatively, evaluation only can be performed on our pretrained models which can be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1VlPQg8KQ2tQAx3xohE1Mq5YqyI9a2EdW?usp=drive_link).

### How to invoke the script

The script can be invoked from the command line like any other python script. It expects 4 arguments of type string to be passed however which are the following:

**Argument 1: baseModel** It can either be *arabert*, *bertweet*, *custom* or *roberta* depending on which model one wants to run.

**Argument 2: language** It can either be *arabic*, *english* or *translated*. Note that arabic can only be used in conjunction with bertweet, as this corresponds to our baseline we compare ourselves to based on true arabic data. The other two options, english and translated, apply with respect to our own trained models on the base English corpus only when choosing english or alternatively on the finetuned version on translated arabic data when choosing arabic.

**Argument 3: mode** It can either be *train* or *eval*. The choice depends on whether one wants to run the whole training process with a subsequent evaluation run or evaluation on a pertrained model only.

**Argument 4: recording** It can either be *True* or any other string argument which logically evaluates to *False*. This argument is only useful when trying to run the training and or evaluation procedure on a subset of batches which was mainly used to record the screencasts for the submission of our project.

### Sample usage

If one wants to run training on the roberta model with a custom classifier on the entire English base corpus, one would have to invoke the program as follows:

```
python3 call_interface.py "custom" "english" "train" "False"
```
The code for launching the pretrained version of the same model in evaluation only mode on a subset of batches would be:
```
python3 call_interface.py "custom" "english" "eval" "True"
```


## Screencast

We provide two different screencasts which should cover all the relevant aspects when taken together. The two screencasts show the following respectively:

- **Screencast_deep_learning_2024_group7_train_eval.webm** shows training and a subsequent evaluation run for each of our models. Note that we perform both tasks only on a subset of 4 batches and training only runs for two epochs, as the screencast would become excessively long otherwise. Moreover, the evalation scores do not match those of the report but such as to reproduce them, we provide the other screencast.

- **Screencast_deep_learning_2024_group7_reproduce_results.webm** reproduces the results presented in the report for each of our models. It simply shows each of the pretrained models being evaluated on the entire respective test set. Please note that the video has been sped up by a factor of 16 since evaluation takes roughly 1 hour and 20 minutes in real time.