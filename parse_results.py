import pandas as pd
import glob


def spot_best_results():
    """Check best consensus score among result files, display results"""

    # loop over results file
    score_max = 0
    for result_file in glob.glob("results/alllm*.csv"):
        df = pd.read_csv(result_file)
        array = result_file.split("/")[-1].split("_")
        model_name = array[-2]
        treshold = array[-1].replace(".csv", "")
        for index, row in df.iterrows():
            if row['EVALUATEUR'] == 'CONSENSUS':
                score = float(row['F1-SCORE'])
                if score >= score_max:
                    score_max = score
                    best_model = model_name
                    best_treshold = treshold

    # display results
    print(f"[BEST MODEL] -> {best_model}")
    print(f"[BEST TRESHOLD] -> {best_treshold}")
    print(f"[BEST CONSENSUS F1] -> {score_max}")

    


if __name__ == "__main__":

    spot_best_results()
    
