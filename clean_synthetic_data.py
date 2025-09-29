import pandas as pd
from Bio import Entrez

def clean_raw_data():
    """reformat text to proper csv"""

    data_file = open("data/synthetic_raw.csv", "r")
    output_data = open("data/synthetic_cleaned.csv", "w")
    for l in data_file:
        lc = l.rstrip().replace('|', ',')
        lc = lc.replace(' ', '')
        output_data.write(f"{lc}\n")
    data_file.close()
    output_data.close()

def add_text():
    """Use PMID to access to title and abstract, save an enrich data file"""

    # init connection
    Entrez.email = "drfox@gmail.com"
    
    data = []
    df = pd.read_csv("data/synthetic_cleaned.csv")
    for index, row in df.iterrows():

        # extract data
        pmid = row['PMID']
        
        handle = Entrez.efetch(db="pubmed", id=str(pmid), retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        try:
            article = records['PubmedArticle'][0]['MedlineCitation']['Article']
            title = article.get('ArticleTitle', '')
            abstract_list = article.get('Abstract', {}).get('AbstractText', [])
            abstract = ' '.join(abstract_list)
        except:
            title = 'NA'
            article = 'NA'


        # craft vector
        vector = {
            "PMID":pmid,
            "title":title,
            "abstract":abstract,
            "adverse_observed":row["adverse_observed"],
            "treatment_safety":row["treatment_safety"],
            "risk_factor":row["risk_factor"],
            "drug_effect":row["drug_effect"],
            "animal_model":row['animal_model']

        }

        # update data
        if title != 'NA' and abstract != 'NA':
            data.append(vector)


    df = pd.DataFrame(data)
    df.to_csv("data/synthetic_enriched.csv", index=False)


if __name__ == "__main__":

    # clean_raw_data()
    add_text()
