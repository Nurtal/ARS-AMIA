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
                       "CM":"R2",
                       "Codage définitif":"LABEL",
                       "ID,TITLE":"SENTENCE",
                       "Titre":"SENTENCE",
                       "Titre ":"SENTENCE"
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

        
    




if __name__ == "__main__":

    clean_title_data()


    
