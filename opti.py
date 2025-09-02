import dspy
from typing import Literal
import pandas as pd
from sklearn.metrics import f1_score

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
    """ """

    # load data
    y = []
    prediction = []
    df = pd.read_csv('data/ressources/optival.csv')
    for index, row in df.iterrows():

        # extract infos
        sentence = row['SENTENCE']
        label = row['LABEL']

        # compute prediction
        pred = 0
        try:
            ad = detect_adverse_effect(sentence)
            if ad['effect'] == 'adverse effect':
                pred = 1
        except:
            pass

        # update vectors
        y.append(label)
        prediction.append(pred)

    # compute F1 score
    score = f1_score(y, prediction)

    return score



if __name__ == "__main__":

    # param llm
    lm = dspy.LM(f"ollama_chat/qwen3:0.6b", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    
    # call evaluation
    m = evaluate_adverse_effect_detection()
    print(f"F1-SCORE => {m}")
