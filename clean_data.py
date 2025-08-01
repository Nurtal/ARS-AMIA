import pandas as pd
import os
import glob


def clean_title_data():
    """Clean dirty xlsx files and turn them in proper csv file, this function handle only title data files"""

    # loop over title data files
    file_name_list = glob.glob("data/raw/*_title.xlsx")
    for file_name in file_name_list:

        # init output repo
        if not os.path.isdir('data/clean'):
            os.mkdir('data/clean')

        # load data
        df = pd.read_excel(file_name)

        # rename columns
        df = df.rename(columns={
                       "Titres":"SENTENCE",
                       "APR":"R1",
                       "CLE":"R2",
                       "CM":"R3",
                       "Codage définitif":"LABEL",
                       "ID,TITLE":"SENTENCE",
                       "Titre":"SENTENCE",
                       "Titre ":"SENTENCE",
                       "Réconciliation":"LABEL"
                   })
        

        # clean SENTENCE
        old_to_new = {}
        for s in list(df['SENTENCE']):
            n = s.split(",",1)
            if len(n) > 1:
                n = n[1]
            else:
                n = n[0]
            old_to_new[s] = n
        df['SENTENCE'] = df['SENTENCE'].replace(old_to_new)

        # save in clean data
        clean_file_name = file_name.split("/")[-1]
        clean_file_name = f"data/clean/{clean_file_name}".replace(".xlsx", ".csv")
        df.to_csv(clean_file_name, index=False)

        

def clean_sentence_data():
    """Clean dirty xlsx files and turn them in proper csv file, this function handle only sentences data files"""

    # loop over sentences data files
    file_name_list = glob.glob("data/raw/*_sentences.xlsx")
    for file_name in file_name_list:

        # init output repo
        if not os.path.isdir('data/clean'):
            os.mkdir('data/clean')

        # load data
        df = pd.read_excel(file_name)

        # rename columns
        df = df.rename(columns={
                       "DOI,TEXT":"SENTENCE",
                       "TEXT":"SENTENCE",
                       "Phrases":"SENTENCE",
                       "Phrases ":"SENTENCE",
                       "APR":"R1",
                       "CLE":"R2",
                       "CM":"R3",
                       "Codage définitif":"LABEL",
                       "ID,TITLE":"SENTENCE",
                       "Titre":"SENTENCE",
                       "Titre ":"SENTENCE",
                       "Réconciliation":"LABEL",
                       "Codage CLE":"R1",
                       "Codage APR":"R2",
                       "Codage CM":"R3",
                       "Codage  CM":"R3",
                       "Codage final":"LABEL",
                       "Codage AP":"R2"
                   })

        # control presence of col name
        all_cols_ok = True
        cols_to_keep = ['SENTENCE','R1','R2','R3','LABEL']
        for cols in cols_to_keep:
            if cols not in list(df.keys()):
                all_cols_ok = False
        if all_cols_ok:
            df = df[cols_to_keep]

            # clean SENTENCE
            old_to_new = {}
            for s in list(df['SENTENCE']):
                n = s.split(",",1)
                if len(n) > 1:
                    n = n[1]
                else:
                    n = n[0]
                old_to_new[s] = n.replace("\n", " ").replace("  ", " ")
            df['SENTENCE'] = df['SENTENCE'].replace(old_to_new)

            # save in clean data
            clean_file_name = file_name.split("/")[-1]
            clean_file_name = f"data/clean/{clean_file_name}".replace(".xlsx", ".csv")
            df.to_csv(clean_file_name, index=False)
        
        else:
            print(f"[CLEAN] -> drop file {file_name}, could not parse columns")
            




if __name__ == "__main__":

    # clean_title_data()
    clean_sentence_data()


    
