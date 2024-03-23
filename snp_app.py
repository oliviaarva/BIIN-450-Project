# Olivia Arva
# Advanced Bioinformatics Project
# SNP Model Application

# Loading in libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


def predict_predisp_snp(data):
    '''
    Using logistic regression model to predict probability of an
    individual being predisposed to CD based on given SNP.
    '''
# Features we want: Allele 1, Allele 2, Effect, P-value, rs, Amino_Acids, 
# Most_severe_consequence, Gene_symbol

# Dropping any NA values
    data = data
    data = data.dropna()

# Obtaining a subset of data for ML model
    subset_data = data[['Allele1', 'Allele2', 'Effect', 'P-value', 'rs',
                        'Amino_Acids', 'Most_severe_consequence',
                        'Gene_symbol']]

# Modify the Gene_symbol feature to contain only one gene name
# Obtains the gene with the highest occurence in each array for each row
    for i, gene_array in enumerate(subset_data['Gene_symbol']):
        # Removing quotes from each value in each array
        gene_array = gene_array.replace('"', '')
        subset_data.at[i, 'Gene_symbol'] = gene_array

    def get_most_common_gene(gene_array):
        gene_array = eval(gene_array.replace('"', ''))
        most_freq_gene = pd.Series(gene_array).value_counts().idxmax()
        return most_freq_gene

    subset_data['Gene_symbol'] = subset_data['Gene_symbol'].apply(get_most_common_gene)
    return subset_data

# Only keeping values in Allele 1 and Allele 2 that occur frequently in dataset
    # freq_allele = ['a', 't', 'c', 'g']
    # filtered_data = subset_data[subset_data['Allele1'].isin(freq_allele)
                               #  & subset_data['Allele2'].isin(freq_allele)]

# One hot encoding categorical columns

    # subset_data = pd.get_dummies(subset_data, columns=['Allele1', 'Allele2'])
    # print(filtered_data['Allele2'].value_counts())
   # print(subset_data.head(5))
    
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
    data = pd.read_csv('CD_data_small.csv')
    # print(data.head(5))
    predict_predisp_snp(data)



if __name__ == "__main__":
    main()