import pandas as pd
import logging

logfilename = "NLP_01_Minimum_Edit_distance/min_edit_distance_log.log"

logging.basicConfig(level=logging.INFO, filename=logfilename, filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Contstants
# STRING1 = "intention" # from string
# STRING2 = "execution" # to string
STRING1 = "manager" # from string
STRING2 = "leader" # to string

LEVENSHTEIN = 2

class Text_Game():
    def __init__(self):
        self.string1 = STRING1
        self.string2 = STRING2

    def calculate_min_edit_distance(self):
        """
        source string X of length n, and target string Y of length m
        X[1..i] and Y[1.. j]

        D[i,j] =
                min (
                   D[i-1,j] + del-cost(source[i])
                   D[i,j-1] + ins-cost(source[i])
                   D[i-1,j-1] + sub-cost(source[i])
                )
        del-cost(source[i] - cost of deletion, 1
        ins-cost(source[i] - cost of insertion, 1
        ins-cost(source[i] - cost of substitution, 2 if Levenshtein

        """
        columns = ['0']+[*self.string1]
        index = ['0']+[*self.string2]
        df = pd.DataFrame(columns=columns, index=index)

        # Transfrom empty string into a new string (column)
        # or existing string to an empty string (row)
        # First row and first column are from 0 to number of letters - deletions
        # First columns are from 0 to number of letters to insert
        df.iloc[0,:] = pd.Series(range(len(STRING1)+1))
        df.iloc[:,0] = pd.Series(range(len(STRING2)+1))

        for j, m in enumerate(self.string2):
            for i, n in enumerate(self.string1):
                logging.info(f"String 1 '{self.string1}': index i = {i},letter {n}")
                logging.info(f"String 2 '{self.string2}': index j = {j},letter {m}")

                if n == m:
                    sub_cost = 0
                else :
                    sub_cost = LEVENSHTEIN

                logging.info(f'substitution i,j : {i},{j} : {df.iloc[j,i]}')
                logging.info(f'deletion i+1,j : {i+1},{j}: {df.iloc[j, i+1]}')
                logging.info(f'insertion i,j+1 : {i},{j+1}: {df.iloc[j+1,i]}')

                rule1 = df.iloc[j+1,i] + 1 # insertion
                rule2 = df.iloc[j, i+1] + 1 # deletion
                rule3 = df.iloc[j,i] + sub_cost # substitution

                # | replace | delete
                # | insert | you are here

                logging.info(f'Rules : {rule3}, {rule2}, {rule1}')

                rule = min(rule1, rule2, rule3)
                df.iloc[j+1, i+1] = rule

                logging.info(df)
                logging.info('-'*30)

        print(f"Edit distance '{self.string1}' -> '{self.string2}' is {df.iloc[-1,-1]}.")




