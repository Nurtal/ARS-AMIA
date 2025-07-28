import dspy
from typing import Literal
import pandas as pd
from sklearn.metrics import f1_score
import os
import glob

class Classify(dspy.Signature):
    """Classify adverse effect / no adverse effect from a given sentence."""

    sentence: str = dspy.InputField()
    effect: Literal["adverse effect", "no adverse effect"] = dspy.OutputField()
    confidence: float = dspy.OutputField()


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


def evaluate_adverse_effect_detection():
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

    # load data
    for data_file in glob.glob("data/clean/*.csv"):
        df = pd.read_csv(data_file)
        df = df[['SENTENCE', 'R1', 'R2', 'R3', 'LABEL']]
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
            y = detect_adverse_effect(sentence)
            if y['effect'] == 'adverse effect':
                predictions.append(1)
            else:
                predictions.append(0)

    # compute F1 - scores
    r1_score = f1_score(r1_label, predictions)
    r2_score = f1_score(r2_label, predictions)
    r3_score = f1_score(r3_label, predictions)
    c_score = f1_score(c_label, predictions)

    # save results
    if not os.path.isdir('results'):
        os.mkdir('results')
    output_file = open("results/scores.csv", "w")
    output_file.write("EVALUATEUR,F1-SCORE\n")
    output_file.write(f"R1,{r1_score}\n")
    output_file.write(f"R2,{r2_score}\n")
    output_file.write(f"R3,{r3_score}\n")
    output_file.write(f"CONSENSUS,{c_score}\n")
    output_file.close()
    



if __name__ == "__main__":

    # configure llm
    lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)

    
    # x = detect_adverse_effect("NO advesre effect detected for this medication")

    evaluate_adverse_effect_detection()
