#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'Overview of data in Excel file'

#row 1 = legend and time stamps for data
#row 2 = legend for data (yield, Mw, Mn, PDI, etc.)

#column 2 (from row 3): sample determiner (consists of MRG-046-G-A(experiment determiner)-letter(used RAFT-Agent)-number(used monomer)-abbreviation(used solvent))
#column 3(from row 3): used reactor for reaction (1 to 15)

#for t=0h

#column 4(from row 3): Mn from SEC (g/mol)
#column 5(from row 3): Mw from SEC (g/mol)
#column 6(from row 3): PDI from SEC (without unit)
#column 7(from row 3): severe tailing or fronting (True or False)
#column 8(from row 3): second distribution or multiple peaks(True or False)
#column 9(from row 3): peak smears into solvent or monomer peak (true or false))
#column 10(from row 3): drop below zero occurred (true or false)(if no data is available, then the data was evaluated by hand)))
#column 11(from row 3): double check was necessary (True or False) (if no data is available, then the data was evaluated by hand))
#column 12(from row 3): Standard substance used for NMR determination of yield (Anisole or Trioxane)
#column 13(from row 3): Peakrange (range or one number (if only one number, the automated tool by Julian Kimmig was used))
#column 14(from row 3): Integral of the Peak assigned with the peakrange before (standard substance signal set to integral of 3 (anisole) or 6 (trioxane)))
#column 15(from row 3): NMR determined Yield 

#for t=1h
#column 16(from row 3): Mn from SEC (g/mol)
#column 17(from row 3): Mw from SEC (g/mol)
#column 18(from row 3): PDI from SEC (without unit)
#column 19(from row 3): severe tailing or fronting (True or False)
#column 20(from row 3): second distribution or multiple peaks(True or False)
#column 21(from row 3): peak smears into solvent or monomer peak (true or false))
#column 22(from row 3): drop below zero occurred (true or false)(if no data is available, then the data was evaluated by hand)))
#column 23(from row 3): double check was necessar (True or False) (if no data is available, then the data was evaluated by hand))
#column 24(from row 3): Integral of the Peak assigned with the peakrange before (standard substance signal set to integral of 3 (anisole) or 6 (trioxane)))
##column 25(from row 3):NMR determined Yield 


#... (until t=15h) (with differences for e.g. t = 6h (only NMR sampling))


#column 82(from row 3): date of experiment
#column 83(from row 3): name of used reactor combination (reflux, reactor assembly, control unit)
#colum 84(from row 3): comments (if necessary)
#column 85(from row 3): reactor underfilled (from 0 to 2) (0 = not underfilled, 1 = underfilled (max. 20%), 2 = underfilled (more than 20%))
#column 86(from row 3): solution in reactor has changed color in comparison to beginning of Experiment (0,1,2,3) (0 = no color change, 1 = decolorized, 2 = slight color change, 3 = strong color change)
#column 87(from row 3): solution in reactor is cloudy (0,1 --> False or True)
#column 88(from row 3): precipitate in reactor (0,1 --> False or True)
#columnn 89(from row 3): "hood" on top of reactor (phase separation) (0,1 --> False or True)
#column 90(from row 3): content of reactor gelated (bulk, not hood or precipiate) (0,1,2 --> 0 = no, 1 = slightly/partly, 2 = fully))
#column 91(from row 3): use data for AI (0,1 --> False or True) (if empty, the decision should be made by this programme)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


######################################################

#Start of the programme

######################################################



# 1. Import of necessary packages

from tkinter import N
from types import AsyncGeneratorType
import numpy as np 
from numpy import negative, rec
import pandas as pd
import os
import re

#######################################################

# 2. Definition of the input and output file path for the Excel documents
INPUT_FILE_PATH = "2023_07_07 - evaluation table (NMR and SEC).xlsx"
OUTPUT_FILE_PATH = "C:\\Users\\xo37lay\\Desktop\\2023_07_07 - evaluation table (NMR and SEC)_temp.xlsx"


# 3. read data from excel file to pd dataframe
df = pd.read_excel(INPUT_FILE_PATH) #reads excel file to pandas dataframe


#######################################################

# 4. restructure the dataframe
   #4.1. Basic changes 
df.dropna(inplace = True, how = "all") #drop all completely empty rows
df = df.drop(df.columns[0],axis=1) #drop first column (legend in the excel file)
df.reset_index(drop=True, inplace=True) #reset index after row removing


    # 4.2. change header of dataframe to more readable names (recently in header only t0h, t1h, t2h, etc. or unnamed)-->(e.g., t0h-Mn)
  
        #4.2.1. replace header (first row in original document contained only information about time of sampling) with the second row (information about the data in the column (without time of sampling)))
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #remove header row from the dataframe
df.columns = new_header #set data from the initial first row as header
df.columns = df.columns.map(lambda x: re.sub(r'\n', '', x)) #remove all line breaks in column names

        #4.2.2. add information about time to the header (e.g. t0h-Mn)
times_list = ["t0h", "t1h", "t2h", "t4h", "t6h", "t8h", "t10h", "t15h"] #list of all times of sampling
new_col_names =[]
times_list_idx = 0

           # 4.2.2.1 iterate over all column names
for col in df:
    # If none of the keywords is found in the column name, change the column name and append it to the new column names list
    if not any(keyword in col for keyword in ['sample determiner', 'reactor', 'date', 'solution', 'data', 'comments', 'Standard', 'Peakrange']):
        
        col_new = times_list[times_list_idx] + "-" + col
        
        if col_new in new_col_names:
            times_list_idx += 1
            col_new = times_list[times_list_idx] + "-" + col
            new_col_names.append(col_new)           
            
        else:
            new_col_names.append(col_new)

    #If one of the keywords is in the column name, do not change the column name and append it to the new column names list
    else:
        col_new = col
        new_col_names.append(col_new)

         # 4.2.2.2 Rename all columns with the new column names
df.columns = new_col_names


        #4.3. split first column (sample determiner) into 4 columns (experiment determiner, RAFT-Agent, monomer, solvent) but keep the first column (sample determiner) in the dataframe
            #4.3.1. drop all rows with empty first column ('sample determiner')
df = df.dropna(subset=[df.columns[0]], how='all')
            #4.3.2. split the first column into 7 columns, rename the columns, add them to the dataframe (and drop the first column (sample determiner))
split_data = df['sample determiner'].str.split('-', expand=True)
split_data.columns = ['Abbreviation','experiment number', 'experiment subnumber', 'experiment determiner', 'monomer','RAFT-Agent', 'solvent']
df = pd.concat([split_data, df], axis =1, sort=False)

            #4.3.3. add a new column (experiment number) to the dataframe (combination of abbreviation, experiment number and experiment subnumber), move it to the first column and drop the columns (abbreviation, experiment number, experiment subnumber)
df['Experiment number'] = df['Abbreviation'].str.cat([df['experiment number'], df['experiment subnumber']], sep='-')
df = df[['Experiment number'] + [col for col in df.columns if col != 'Experiment number']]
df = df.drop(columns = ['Abbreviation', 'experiment number', 'experiment subnumber'])

            #4.3.4. add new column for later comparison with all possible permutations (combination of monomer, RAFT-Agent and solvent)
df['possible sample determiner-original'] = df['monomer'].str.cat([df['RAFT-Agent'], df['solvent']], sep='-')

        #4.4. remove all trailing spaces from the column names  
df.columns = df.columns.str.strip()



###################################################################

# 5. curation criteria

        #5.0 Remove all values which are NaN to np.nan
df.replace('NaN', np.nan, inplace=True)
df.replace('NA', np.nan, inplace=True)    
df.replace('na', np.nan, inplace=True)
df.replace('N/A', np.nan, inplace=True)
df.replace('n/a', np.nan, inplace=True)


        #5.1. if column 'use data for AI' is marked with a 0 , discard data and add data to a new data frame
discarded_df = df[df['use data for AI'] == 0]
discarded_df = discarded_df.reset_index(drop=True)
df.drop(df[df['use data for AI'] == 0].index, inplace = True)
df = df.reset_index(drop=True)


        #5.2. if reactor underfilled == 2, remove row and add row to new dataframe which is later printed to excel file as discarded samples
discarded_df2 = df[df['reactor is underfilled after polymerization?'] == 2]
new_df = pd.concat([discarded_df, discarded_df2])
discarded_df = new_df
discarded_df.reset_index(drop=True)
df.drop(df[df['reactor is underfilled after polymerization?'] == 2].index, inplace = True)
df = df.reset_index(drop=True)


        #5.3. replace column values (e.g. if column value is ooc or for Mn: > 100.000 g/mol, replace with NaN)
            #5.3.1. define regular expressions for the columns which should be checked (e.g. all Mn or Mw or yield columns)
regex_Mn = r't\d+h-Mn'
regex_Mw = r't\d+h-Mw'
regex_dispersity = r't\d+h-\u00d0'
regex_yield = r't\d+h-yield'
            
            #5.3.2. for Mn, Mw remove ooc and replace > 100.000 g/mol and 0 g/mol with NaN
                # Filter columns that match the pattern
filtered_columns_molar_mass = [col for col in df.columns if re.match(regex_Mn, col) or re.match(regex_Mw, col)]
                # Iterate over the matched columns and rows
for column_name in filtered_columns_molar_mass:
    for i in range(len(df)):
        value = df[column_name][i]
                    # Check if the value is a string and contains the pattern '\s*ooc\s*', if so: replace with NaN
        if isinstance(value, str) and re.search(r'ooc', value):
            df[column_name] = df[column_name].replace(value, np.nan)
                    #Check if value is a float and larger than 100000, if so: replace with NaN
        elif float(value) > 100000:
            df[column_name] = df[column_name].replace(value, np.nan)
     
        
''' Activation maybe later (if necessary)

            #5.3.3. for dispersity remove > 2.2 with NaN and also remove the corresponding Mn and Mw values 
            #(the Mn and Mw parts still need to be implemented)
                # Filter columns that match the pattern
filtered_columns_molar_mass = [col for col in df.columns if re.match(regex_dispersity, col)]
                # Iterate over the matched columns and rows
for column_name in filtered_columns_molar_mass:
    for i in range(len(df)):
        value = df[column_name][i]
                    #Check if value is a float and larger than 2.2, if so: replace with NaN
        if float(value) > 2.2:
            df[column_name] = df[column_name].replace(value, np.nan)            
'''


            #5.3.4. for yield remove negative yields and yields which are negative in comparison to previous timepoint by at least 10% (Ungenauigkeit der Methode)
                #5.3.4.1. remove negative yields (< -5%)
filtered_columns_yield = [col for col in df.columns if re.match(regex_yield, col)]
for column_name in filtered_columns_yield:
    for i in range(len(df)):
        value = df[column_name][i]
        #check if yield is negative (larger than 5% negative)
        if float(value) < -0.05:
            df[column_name] = df[column_name].replace(value, np.nan)
                
                #5.3.4.2. check if yield is negative in comparison to previous timepoint by at least 10% (Ungenauigkeit der Methode)
                    # Define the threshold for the 10% decrease for consecutive time points --> more than 10% decrease in comparison to previous timepoint would lead to NaN
threshold = 0.1   

                        # Iterate through the rows and perform the comparisons
for i in range(1, len(df)):
    for col in range(1,len(filtered_columns_yield)):
       current_column = filtered_columns_yield[col]
       previous_column = filtered_columns_yield[col - 1]
       current_value = df.at[i, current_column]
       previous_value = df.at[i, previous_column]
                            #check if both values for comparison are floats, then compare them
       if isinstance(current_value, float) and isinstance (previous_value, float):
            if current_value < (previous_value - threshold):
                df.at[i, current_column] = np.nan
       else:
            continue
            
        
            #5.3.5. remove all datasets (rows) which have less than x (x=4) full (Mn,Mw, D) SEC data points and/or NMR data points(yields)
REMOVER_DECIDER = 4     #number of data points which are necessary to keep the data set
rows_to_remove = []     #list of rows which should be removed


                #5.3.5.1. iterate through the rows  
for index, row in df.iterrows():
    is_complete_SEC = 0
    is_complete_NMR = 0
    
                    #5.3.5.1.1. iterate through the time points for the SEC data points
                        #if one of the data points is NaN, add 0 to the is_complete variable
    for time_point in times_list:
        if pd.isna(row[time_point + '-Mn']):
            is_complete_SEC += 0
        
        elif pd.isna(row[time_point + '-Mw']):
            is_complete_SEC += 0    
     
        elif pd.isna(row[time_point + '-\u00d0']):
            is_complete_SEC += 0
                        #if all data points are not NaN, add 1 to the is_complete variable
        else: 
            is_complete_SEC += 1
        
                    #5.3.5.1.2. same as above for NMR data points
    for time_point in times_list:
                        #5.3.5.1.2.1. use all time points except t6h and t10h (only SEC sampling for those)
        if time_point != 't6h' and time_point != 't10h': 

             if pd.isna(row[time_point + '-yield']):
                  is_complete_NMR += 0
             else:
                 is_complete_NMR += 1
            
                #5.3.5.2 check if the number of complete SEC and NMR data points is smaller than the REMOVER_DECIDER
    if is_complete_NMR < REMOVER_DECIDER or is_complete_SEC < REMOVER_DECIDER:
        rows_to_remove.append(index)
      
                #5.3.5.3. Remove the rows from df and add them to discarded_df
discarded_df = df.loc[rows_to_remove].copy()
df = df.drop(rows_to_remove)

                #5.3.5.4. Reset the indices of the dataframes
df = df.reset_index(drop=True)
discarded_df = discarded_df.reset_index(drop=True)

###################################################################

#6. check which data is still missing to have performed at least each experiment once
import itertools

    #6.1. Define the options for the three variables in the experiment 
RAFTagent_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
monomer_options = ['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
solvent_options = ['DMF', 'DMSO', 'Tol']

    #6.2. Create a list to store the permutations
permutations = []

    #6.3. Generate permutations for the options
for RAFTagent_options, monomer_options, solvent_options in itertools.product(RAFTagent_options, monomer_options, solvent_options):
    # Build the new string with the chosen options
    new_string = f'{monomer_options}-{RAFTagent_options}-{solvent_options}'
    permutations.append(new_string)
 
    #6.4. Create a dataframe from the list of permutations
permutations_df = pd.DataFrame(permutations, columns = ['possible sample determiner-permutations'])

    #6.5. Compare the permutations dataframe with the curated dataframe from the excel file
       #6.5.1. Check if the permutations are part of the existing dataframe
       #       if so, delete them from the permutations dataframe, which is later printed to the excel file to see which experiments still need to be conducted 
permutations_df.drop(permutations_df[permutations_df['possible sample determiner-permutations'].isin(df['possible sample determiner-original'])].index, inplace = True)

###################################################################

# 7. save the dataframe to excel file
    #7.1. # create an excel writer object
with pd.ExcelWriter(OUTPUT_FILE_PATH) as writer:
        # use to_excel function and specify the sheet_name and index 
        # to store the dataframe in specified sheet
    df.to_excel(writer, sheet_name='utilizable samples', index=False)
    discarded_df.to_excel(writer, sheet_name="discarded samples", index=False)
    permutations_df.to_excel(writer, sheet_name="Missing experiments", index=False)
   