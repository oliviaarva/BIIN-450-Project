# Olivia Arva
# Advanced Bioinformatics Project
# SNP Model Application

# Loading in packages and libraries
import ast
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


def cleaning_data(file):
    '''
    Filtering and cleaning the data to make it suitable for
    the ML algorithm.
    '''
    # Dropping any NA values
    full_data = file.dropna()

    # Obtaining a subset of data for ML model
    data = full_data[['Allele1', 'Allele2', 'Effect', 'P-value', 'rs',
                      'StdErr', 'Most_severe_consequence',
                      '']]

    # Only keeping values in Allele 1 and Allele 2 that occur freq in dataset
    freq_allele = ['a', 't', 'c', 'g']
    data = data[data['Allele1'].isin(freq_allele)
                & data['Allele2'].isin(freq_allele)]

    '''
    # Modify the Gene_symbol feature to contain only one gene name
    # Obtains the gene with the highest occurence in each array for each row
    for i, gene_array in enumerate(data['Gene_symbol']):
        try:
            # Converting string to list of object values
            gene_list = ast.literal_eval(gene_array)
        except ValueError:
            # If ast.literal_eval cannot evaluate the array
            gene_array = gene_array.split('[', '').replace(']', '')
            gene_list = gene_array.split(',')
            gene_list = [gene.strip(' " ') for gene in gene_list]

        # Counting and selecting gene with highest occurence
        gene_count = Counter(gene_list)
        most_common_gene = max(gene_count, key=gene_count.get)

        data.at[i, 'Gene_symbol'] = most_common_gene
        '''
    return data
    



def predict_predisp_snp(data):
    '''
    Using logistic regression model to predict probability of an
    individual being predisposed to CD based on given SNP.
    '''
    
# Obtaining the labels (what we want to predict) and features
    '''
    features = data.drop('Effect', axis=1)
    labels = data['Effect']

# Splitting the dataset 80:20 into features and labels
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    
# Calling the logistic regression model
    model = LogisticRegression()

# Fitting the model on the training features and label
    model.fit(features_train, labels_train)
'''
# Using statistical analysis on model


# Loading in the dataframe
def main():
    file = pd.read_csv('CD_data_small.csv')
    # print(file.head(5))

    data = cleaning_data(file)
    predict_predisp_snp(data)



if __name__ == "__main__":
    main()