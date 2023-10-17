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

import pandas as pd
import os
import re

#######################################################

# 2. Definition of the input and output file path for the Excel documents
INPUT_FILE_PATH = "C:\\Users\\xo37lay\\Desktop\\2023_07_07 - evaluation table (NMR and SEC).xlsx"
OUTPUT_FILE_PATH = "C:\\Users\\xo37lay\\Desktop\\2023_07_07 - evaluation table (NMR and SEC)_temp.xlsx"



# 3. read data from excel file to pd dataframe
df = pd.read_excel(INPUT_FILE_PATH) #reads excel file to pandas dataframe


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

        # 4.2.2.2 rename all columns with the new column names
df.columns = new_col_names




        #4.3. split first column (sample determiner) into 4 columns (experiment determiner, RAFT-Agent, monomer, solvent) but keep the first column (sample determiner) in the dataframe
            #4.3.1. drop all rows with empty first column ('sample determiner')
df = df.dropna(subset=[df.columns[0]], how='all')
            #4.3.2. split the first column into 7 columns, rename the columns, add them to the dataframe and drop the first column (sample determiner)
split_data = df['sample determiner'].str.split('-', expand=True)
split_data.columns = ['Abbreviation','experiment number', 'experiment subnumber', 'experiment determiner', 'monomer','RAFT-Agent', 'solvent']
df = pd.concat([split_data, df], axis =1, sort=False)
df =df.drop(columns = 'sample determiner')
            #4.3.3. add a new column (experiment number) to the dataframe (combination of abbreviation, experiment number and experiment subnumber), move it to the first column and drop the columns (abbreviation, experiment number, experiment subnumber)
df['Experiment number'] = df['Abbreviation'].str.cat([df['experiment number'], df['experiment subnumber']], sep='-')
df = df[['Experiment number'] + [col for col in df.columns if col != 'Experiment number']]
df = df.drop(columns = ['Abbreviation', 'experiment number', 'experiment subnumber'])


#einfügen der Kurationskriterien und speichern der gelöschten Reihen (zumindest mit Namen in einem neuen Excel Arbeitsblatt)


# 5. save the dataframe to excel file

df.to_excel(OUTPUT_FILE_PATH, index=False)