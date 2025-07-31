import dspy
from typing import Literal
import pandas as pd
from sklearn.metrics import f1_score
import os
import glob
import re


#-------------------------------#
# LLM & DSPY clacc and concepts #
#-------------------------------#

# class based on dspy
class Classify(dspy.Signature):
    """Classify adverse effect / no adverse effect from a given sentence."""

    sentence: str = dspy.InputField()
    effect: Literal["adverse effect", "no adverse effect"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


class AnimalModel(dspy.Signature):
    """Detect if sentence refer to an animal model"""

    sentence: str = dspy.InputField()
    animal: Literal["animal model", "no animal model"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

class TreatmentSafety(dspy.Signature):
    """ """

    sentence: str = dspy.InputField()
    effect: Literal["mention treatment safety", "does not mention treatment safety"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

class RiskFactor(dspy.Signature):
    """ """

    sentence: str = dspy.InputField()
    effect: Literal["mention risk factor", "does not mention risk factor"] = dspy.OutputField()
    confidence: float = dspy.OutputField()

class DrugEffect(dspy.Signature):
    """ """

    sentence: str = dspy.InputField()
    effect: Literal["mention drug effect", "does not mention drug effect"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


# function to call dspy 'classifier'
def detect_adverse_effect(sentence:str) -> dict:
    """Detect adverse effect within a sentence

    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * effect : can be adverse effect / no adverse effect
            * confidence : confidence of the classication
    """
    
    # run llm
    classify = dspy.Predict(Classify)
    result = classify(sentence=sentence)

    # return results
    return {'sentence':sentence, 'effect':result['effect'], 'confidence':result['confidence']}


def detect_treatment_safety(sentence:str) -> dict:
    """Detect treatement safety within a sentence

    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * effect : can be mention treatment safety / does not mention treatment safety
            * confidence : confidence of the classication
    """
    
    # run llm
    classify = dspy.Predict(TreatmentSafety)
    result = classify(sentence=sentence)

    # return results
    return {'sentence':sentence, 'effect':result['effect'], 'confidence':result['confidence']}


def detect_risk_factor(sentence:str) -> dict:
    """Detect risk factor within a sentence

    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * effect : can be mention risk factor / does not mention risk factor
            * confidence : confidence of the classication
    """
    
    # run llm
    classify = dspy.Predict(RiskFactor)
    result = classify(sentence=sentence)

    # return results
    return {'sentence':sentence, 'effect':result['effect'], 'confidence':result['confidence']}


def detect_drug_effect(sentence:str) -> dict:
    """Detect drug effect within a sentence

    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * effect : can be mention drug effect / does not mention drug effect
            * confidence : confidence of the classication
    """
    
    # run llm
    classify = dspy.Predict(DrugEffect)
    result = classify(sentence=sentence)

    # return results
    return {'sentence':sentence, 'effect':result['effect'], 'confidence':result['confidence']}


def detect_animal_model(sentence:str) -> dict:
    """Detect animal model within a sentence

    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * animal : animal model / no animal model
            * confidence : confidence of the classication
    """
    
    # run llm
    classify = dspy.Predict(AnimalModel)

    # very bold call, but since llm output stupid answer, assume it returned the name of the animal model when it
    # does not return something its allowed to return
    try:
        result = classify(sentence=sentence)
    except:
        result = {'animal':'animal model in the sentence', 'confidence':0.51}
        
    # return results
    return {'sentence':sentence, 'animal':result['animal'], 'confidence':result['confidence']}




#---------------------#
# REGEX fast function #
#---------------------#

def detect_animal_model_fast(sentence:str) -> dict:
    """Fast and simple way of detecting animal model in a sentence
    
    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * animal : animal model / no animal model
            * confidence : confidence of the classication (always 1.0 in this case)
    """

    # fast detection
    target_list = ['rabbit', 'mouse', 'mice', ' rats']
    r = 'no animal model'
    for elt in target_list:
        if re.search(elt, sentence):
            r = 'animal model'

    # return results
    return {'sentence':sentence, 'animal':r, 'confidence':1.0}


def detect_adverse_effect_fast(sentence:str) -> dict:
    """Fast way to preshot obvisous sentence
    
    Args:
        - sentence (str) : input data, parsed by the llm

    Return:
        - (dict) : contains the following keys :
            * sentence : input sentence
            * animal : adverse effect / no adverse effect
            * confidence : confidence of the classication (always 1.0 in this case)
     
    """   

    target_list = ['adverse event', 'adverse effect', 'safety']
    r = 'no adverse effect'
    for elt in target_list:
        if re.search(elt, sentence.lower()):
            r = 'adverse effect'

    # return results
    return {'sentence':sentence, 'effect':r, 'confidence':1.0}

def compute_sentence_score(sentence:str) -> float:
    """ """

    # init var
    sentence_selected = 0
    score = 0

    # run llms
    try:
        ad = detect_adverse_effect(sentence) 
    except:
        ad = {'effect':'NA'}
    try:
        ts = detect_treatment_safety(sentence) 
    except:
        ts = {'effect':'NA'}
    try:
        rf = detect_risk_factor(sentence) 
    except:
        rf = {'effect':'NA'}
    try:
        de = detect_drug_effect(sentence)
    except:
        de = {'effect':'NA'}

    # compute score
    if ad['effect'] == 'afverse effect':
        score += 1.0 * ad['confidence'] 
    if ts['effect'] == 'mention treatment safety':
        score += 1.0 * ts['confidence'] 
    if rf['effect'] == 'mention risk factor':
        score += 1.0 * rf['confidence'] 
    if de['effect'] == 'mention drug effect':
        score += 1.0 * de['confidence'] 

    # return score
    return score
        
    

def evaluate_adverse_effect_detection(model_name, score_treshold):
    """Compute F1-score based on adverse effect detection,
    create a score.csv file in results folder with F1-score assiciated
    to comparison with each reviewer and the consensus

    N.B : possible que le score dans les fichiers soit plus complexe que simplement
    la detection d'un adverse effect (e.g drop animal model & co)
    """

    # init labels array
    r1_label = []
    r2_label = []
    r3_label = []
    c_label = []
    predictions = []

    # init result folder
    if not os.path.isdir('results'):
        os.mkdir('results')

    # init miss match log file
    miss_log_file = open("results/miss_match.log", "w")
    miss_log_file.write("SENTENCE,ANIMAL,ADVERSE_EFFECT,PREDICTION,LABEL\n")

    # init score log file
    sentence_score_log = open(f"results/allm_{model_name}_scores.csv", "w")
    sentence_score_log.write(f"SENTENCE,SCORE,PREDICTION,LABEL\n")

    # load data
    for data_file in glob.glob("data/clean/*sentences.csv"):
        df = pd.read_csv(data_file)
        df = df[['SENTENCE', 'R1', 'R2', 'R3', 'LABEL']]

        # deal with str in bad places
        df['R1'] = df['R1'].apply(lambda x: None if isinstance(x, str) else x)
        df['R2'] = df['R2'].apply(lambda x: None if isinstance(x, str) else x)
        df['R3'] = df['R3'].apply(lambda x: None if isinstance(x, str) else x)
        df['LABEL'] = df['LABEL'].apply(lambda x: None if isinstance(x, str) else x)

        df = df.dropna()

        for index, row in df.iterrows():

            # extract infos
            sentence = row['SENTENCE']
            r1 = row['R1']
            r2 = row['R2']        
            r3 = row['R3']        
            label = row['LABEL']

            # control extracted labels
            if r1 in [0, -1]:
                r1 = 0
            if r2 in [0, -1]:
                r2 = 0
            if r3 in [0, -1]:
                r3 = 0
            if label in [0, -1]:
                label = 0

            # extend labels vector
            r1_label.append(r1)
            r2_label.append(r2)
            r3_label.append(r3)
            c_label.append(label)

            # perform prediction
            # detect animal model
            a = detect_animal_model_fast(sentence)
            y = {'effect':'NA'}
            pred = 0
            score = 'NA'
            if a['animal'] != "animal model":

                # call llms
                score = compute_sentence_score(sentence) 

                # make prediction based on score
                if score >= score_treshold:
                    pred = 1

            else:
                pred = 0

            # update predictions
            predictions.append(pred)

            # log miss match
            if pred != label:
                miss_log_file.write(f"{sentence},{a['animal']},{y['effect']},{pred},{label}\n")
            sentence_score_log.write(f"{sentence},{score},{pred},{label}\n")

    # compute F1 - scores
    r1_score = f1_score(r1_label, predictions)
    r2_score = f1_score(r2_label, predictions)
    r3_score = f1_score(r3_label, predictions)
    c_score = f1_score(c_label, predictions)

    # save results
    output_file = open(f"results/alllm_scores_{model_name}_{score_treshold}.csv", "w")
    output_file.write("EVALUATEUR,F1-SCORE\n")
    output_file.write(f"R1,{r1_score}\n")
    output_file.write(f"R2,{r2_score}\n")
    output_file.write(f"R3,{r3_score}\n")
    output_file.write(f"CONSENSUS,{c_score}\n")
    output_file.close()

    # close logs files
    miss_log_file.close()
    



if __name__ == "__main__":

    # configure llm
    lm = dspy.LM("ollama_chat/phi3", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)

    
    # x = detect_adverse_effect("NO advesre effect detected for this medication")

    model_list = [
        'phi3', 
        'llama3.2', 
        'mistral', 
        'qwen3',
        'deepseek-r1',
        'phi4',
        'alfred',
        'gemma3'
    ]

    for m in model_list:
        for x in [0.2,0.3,0.4,0.5,1,1.5,2,2.5,3,3.5,4]:

            # param llm
            lm = dspy.LM(f"ollama_chat/{m}", api_base="http://localhost:11434", api_key="")
            dspy.configure(lm=lm)

            # run
            evaluate_adverse_effect_detection(m, x)
