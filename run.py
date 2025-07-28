import dspy
from typing import Literal

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


if __name__ == "__main__":

    # configure llm
    lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)

    
    x = detect_adverse_effect("NO advesre effect detected for this medication")
