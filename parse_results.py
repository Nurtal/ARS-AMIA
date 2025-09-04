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


def spot_best_agent_results():
    """Parse results of agent benchmark, display best labels """    

    agent_to_score = {}
    agent_to_best_config_id = {}
    for result_file in glob.glob("agent_benchmark/*.csv"):

        # load data
        df = pd.read_csv(result_file)

        # extract infos
        array = result_file.split("/")[-1].split("_")
        agent = array[0]
        config_id = array[2]

        # update dict
        if agent not in agent_to_score:
            agent_to_score[agent] = 0 
        if agent not in agent_to_best_config_id:
            agent_to_best_config_id[agent] = 'NA'

        # evaluate score
        for index, row in df.iterrows():
            if row['EVALUATEUR'] == 'CONSENSUS':
                score = float(row['F1-SCORE'])
                if score >= agent_to_score[agent]:
                    agent_to_score[agent] = score
                    agent_to_best_config_id[agent] = int(config_id)
        
    # load configuration
    for agent in agent_to_best_config_id:

        best_conf = agent_to_best_config_id[agent]
        df_config = pd.read_csv(f"data/ressources/{agent}_labels.csv")
        df_config = df_config[df_config['ID'] == best_conf]

        # extract labels
        label1 = list(df_config['LABEL1'])[0]
        label2 = list(df_config['LABEL2'])[0]

        # display results
        print(f"[AGENT {agent}] BEST CONFIG -> {label1} vs {label2} ({agent_to_score[agent]})")
        

if __name__ == "__main__":

    # spot_best_results()
    spot_best_agent_results()
    
