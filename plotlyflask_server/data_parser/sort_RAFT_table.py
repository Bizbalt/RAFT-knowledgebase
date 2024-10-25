"""

Overview of data in Excel file

row 1 = legend and time stamps for data
row 2 = legend for data (conversion, Mw, Mn, PDI, etc.)

column 2 (from row 3): sample determiner (consists of
 MRG-046-G-A(experiment determiner)
 letter(used RAFT-Agent)
 number(used monomer)
 abbreviation(used solvent))
column 3(from row 3): used reactor for reaction (1 to 15)

for t=0h
column 4(from row 3): Mn from SEC (g/mol)
column 5(from row 3): Mw from SEC (g/mol)
column 6(from row 3): PDI from SEC (without unit)
column 7(from row 3): severe tailing or fronting (True or False)
column 8(from row 3): second distribution or multiple peaks(True or False)
column 9(from row 3): peak smears into solvent or monomer peak (true or false))
column 10(from row 3): drop below zero occurred (true or false)(if no data is available,
 then the data was evaluated by hand)))
column 11(from row 3): double check was necessary (True or False) (if no data is available,
 then the data was evaluated by hand))
column 12(from row 3): Standard substance used for NMR determination of conversion (Anisole or Trioxane)
column 13(from row 3): Peakrange (range or one number
 (if only one number, the automated tool by Julian Kimmig was used))
column 14(from row 3): Integral of the Peak assigned with the peakrange before
 (standard substance signal set to integral of 3 (anisole) or 6 (trioxane)))
column 15(from row 3): NMR determined Yield

for t=1h
column 16(from row 3): Mn from SEC (g/mol)
column 17(from row 3): Mw from SEC (g/mol)
column 18(from row 3): PDI from SEC (without unit)
column 19(from row 3): severe tailing or fronting (True or False)
column 20(from row 3): second distribution or multiple peaks(True or False)
column 21(from row 3): peak smears into solvent or monomer peak (true or false))
column 22(from row 3): drop below zero occurred (true or false)(if no data is available,
 then the data was evaluated by hand)))
column 23(from row 3): double check was necessary (True or False) (if no data is available,
 then the data was evaluated by hand))
column 24(from row 3): Integral of the Peak assigned with the peakrange before
 (standard substance signal set to integral of 3 (anisole) or 6 (trioxane)))
column 25(from row 3):NMR determined Yield


until t=15h (only NMR for t = 6h and t = 10h, no SEC)
column 82(from row 3): date of experiment
column 83(from row 3): name of used reactor combination (reflux, reactor assembly, control unit)
colum 84(from row 3): comments (if necessary)
column 85(from row 3): reactor underfilled (from 0 to 2) (0 = not underfilled, 1 = underfilled
 (max. 20%), 2 = underfilled (more than 20%))
column 86(from row 3): solution in reactor has changed color in comparison to beginning of Experiment
 (0,1,2,3) (0 = no color change, 1 = de-colorized, 2 = slight color change, 3 = strong color change)
column 87(from row 3): solution in reactor is cloudy (0,1 --> False or True)
column 88(from row 3): precipitate in reactor (0,1 --> False or True)
# column 89(from row 3): "hood" on top of reactor (phase separation) (0,1 --> False or True)
column 90(from row 3): content of reactor gelated (bulk, not hood or precipitate)
 (0,1,2 --> 0 = no, 1 = slightly/partly, 2 = fully))
column 91(from row 3): use data for AI (0,1 --> False or True)
 (if empty, the decision should be made by this programme)

"""

# 1. Import statements

import numpy as np
import pandas as pd
import re
import os
import itertools
from datetime import datetime

#######################################################

# 2. Definition of the input and output file path for the Excel documents
INPUT_FILE_PATH = "data/kinetics_data/2023_07_07 - evaluation table (NMR and SEC).xlsx"
OUTPUT_FILE_PATH = os.path.join("data/kinetics_data",
                                str(datetime.today().date()) + " - evaluation table (NMR and SEC)_curated.xlsx")

# 3. read data from Excel file to pd dataframe
df = pd.read_excel(INPUT_FILE_PATH)  # reads Excel file to pandas dataframe

#######################################################

# 4. restructure the dataframe
# 4.1. Basic changes
df.dropna(inplace=True, how="all")  # drop all completely empty rows
df = df.drop(df.columns[0], axis=1)  # drop first column (legend in the Excel file)
df.reset_index(drop=True, inplace=True)  # reset index after row removing

# 4.2. change header of dataframe to more readable names
#  (recently in header only t0h, t1h, t2h, etc. or unnamed)-->(e.g., t0h-Mn)

# 4.2.1. replace header (first row in original document contained only information about time of sampling)
#  with the second row (information about the data in the column (without time of sampling)))
new_header = df.iloc[0]  # grab the first row for the header
df = df[1:]  # remove header row from the dataframe
df.columns = new_header  # set data from the initial first row as header
df.columns = df.columns.map(lambda x: re.sub(r'\n', '', x))  # remove all line breaks in column names

# 4.2.2. add information about time to the header (e.g. t0h-Mn)
times_list = ["t0h", "t1h", "t2h", "t4h", "t6h", "t8h", "t10h", "t15h"]  # list of all times of sampling
new_col_names = []
times_list_idx = 0

# 4.2.2.1 iterate over all column names
for col in df:
    # If none of the keywords is found in the column name, change the column name and append it to the new column
    #  names list
    if not any(keyword in col for keyword in
               ['sample determiner', 'reactor', 'date', 'solution', 'data', 'comments', 'Standard', 'Peakrange']):

        col_new = times_list[times_list_idx] + "-" + col

        if col_new in new_col_names:
            times_list_idx += 1
            col_new = times_list[times_list_idx] + "-" + col
            new_col_names.append(col_new)

        else:
            new_col_names.append(col_new)

    # If one of the keywords is in the column name, do not change the column name and append it to the new column
    # names list
    else:
        new_col_names.append(col)

        # 4.2.2.2 Rename all columns with the new column names
df.columns = new_col_names

# 4.3. split first column (sample determiner) into 4 columns (batch(prior experiment determiner), RAFT-Agent, monomer,
# solvent) but keep the first column (sample determiner) in the dataframe 4.3.1. drop all rows with empty first
# column ('sample determiner')
df = df.dropna(subset=[df.columns[0]], how='all')
# 4.3.2. split the first column into 7 columns, rename the columns, add them to the dataframe
#  (and drop the first column (sample determiner))
split_data = df['sample determiner'].str.split('-', expand=True)
split_data.columns = ['Abbreviation', 'experiment number', 'experiment subnumber', 'batch', 'monomer',
                      'RAFT-Agent', 'solvent']
df = pd.concat([split_data, df], axis=1, sort=False)

# 4.3.3. add a new column (experiment number) to the dataframe (combination of abbreviation, experiment number
#  and experiment sub number), move it to the first column and drop the columns (abbreviation, experiment number,
#   experiment sub number)
df['Experiment number'] = df['Abbreviation'].str.cat([df['experiment number'], df['experiment subnumber']], sep='-')
df = df[['Experiment number'] + [col for col in df.columns if col != 'Experiment number']]  # reorder columns
df = df.drop(columns=['Abbreviation', 'experiment number', 'experiment subnumber'])

#  The experiment number is not a unique identifier and therefore obsolete
df["Experiment number"] = range(1, len(df) + 1)

# 4.3.4. add new column for later comparison with all possible permutations (combination of monomer, RAFT-Agent
#  and solvent)
df['possible sample determiner-original'] = df['monomer'].str.cat([df['RAFT-Agent'], df['solvent']], sep='-')

# 4.4. remove all trailing spaces from the column names
df.columns = [x.strip() for x in df.columns]

###################################################################

# 5. curation criteria

# 5.0 Replace all values which are NaN to np.nan
df.replace('NaN', np.nan, inplace=True)
df.replace('NA', np.nan, inplace=True)
df.replace('na', np.nan, inplace=True)
df.replace('N/A', np.nan, inplace=True)
df.replace('n/a', np.nan, inplace=True)

# 5.1. if column 'use data for AI' is marked with a 0 , discard data and add data to a new data frame
discarded_df = df[df['use data for AI'] == 0].copy()
# Create a new column 'criterion' with a default value in discarded_df
discarded_df['discarding criterion'] = None
discarded_df = discarded_df.reset_index(drop=True)
# Fill the 'criteria' column where 'use data for AI' is 0
discarded_df.loc[discarded_df['use data for AI'] == 0, 'discarding criterion'] = 'use-data-for-AI=0'
# Drop all rows where 'use data for AI' is 0 from original dataframe
df.drop(df[df['use data for AI'] == 0].index, inplace=True)
# reset the index of the original dataframe
df = df.reset_index(drop=True)


# 5.2. if reactor underfilled > 1, remove row and add row to new dataframe which is later printed to Excel
#  file as discarded samples
#   5.2.2 and 5.2.3 do the same with Precipitation and gelation/phase separation in the reactor
criteria_and_threshold = {"reactor is underfilled after polymerization?": 1.1,
                          "Precipitate in reactor?": 0.001,
                          "hood on top of reactor? (phase separation)": 0.001}
abbreviation_criteria = {"reactor is underfilled after polymerization?": "underfilled",
                         "Precipitate in reactor?": "precipitate",
                         "hood on top of reactor? (phase separation)": "phase separation"}

for criteria, threshold in zip(criteria_and_threshold.keys(), criteria_and_threshold.values()):
    _temp_df = df[df[criteria] >= threshold].copy()
    # Create a new column 'criterion' with a default value in discarded_df
    _temp_df['discarding criterion'] = None
    _temp_df = _temp_df.reset_index(drop=True)
    # Fill the 'criteria' column where 'reactor is underfilled after polymerization?' is >= 1
    _temp_df.loc[_temp_df[criteria] >= threshold, 'discarding criterion'] = abbreviation_criteria[criteria]
    new_df = pd.concat([discarded_df, _temp_df])
    discarded_df = new_df
    discarded_df.reset_index(drop=True)
    # Drop all rows where 'reactor is underfilled after polymerization?' is >= 1 from original dataframe
    df.drop(df[df[criteria] >= threshold].index, inplace=True)
    # reset the index of the original dataframe
    df = df.reset_index(drop=True)

# 5.3. replace column values (e.g. if column value is ooc or for Mn: > 100.000 g/mol, replace with NaN)
# 5.3.1. define regular expressions for the columns which should be checked (e.g. all Mn or Mw or conversion columns)
regex_Mn = r't\d+h-Mn'
regex_Mw = r't\d+h-Mw'
regex_dispersity = r't\d+h-\u00d0'
regex_conversion = r't\d+h-conversion'

# 5.3.2. for Mn, Mw remove ooc and replace > 100.000 g/mol and 0 g/mol with NaN
# Filter columns that match the pattern
filtered_columns_molar_mass = [col for col in df.columns if re.match(regex_Mn, col) or re.match(regex_Mw, col)]
# Iterate over the matched columns and rows
for column_name in filtered_columns_molar_mass:
    for index in df.index:
        value = df.loc[index, column_name]
        # Check if the value is a string and contains the pattern '\s*ooc\s*', if so: replace with NaN
        if isinstance(value, str) and re.search(r'ooc', value):
            df.loc[index, column_name] = np.nan
            # Check if value is a float and larger than 100000, if so: replace with NaN
        elif float(value) > 100000:
            df.loc[index, column_name] = np.nan

''' Activation maybe later (if required)

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

# 5.3.4. for conversion remove negative conversions and conversions which are negative in comparison to previous
#  time points by at least 10% (Method inaccuracy)
# 5.3.4.1. remove negative conversions (< -5%)
filtered_columns_conversion = [col for col in df.columns if re.match(regex_conversion, col)]
for column_name in filtered_columns_conversion:
    for index in df.index:
        value = df.loc[index, column_name]
        # check if conversion is negative (larger than 5% negative)
        if float(value) < -0.05:
            df.loc[index, column_name] = np.nan

            # 5.3.4.2. check if conversion is negative in comparison to previous time point by at least 10% (Method
            # inaccuracy) Define the threshold for the 10% decrease for consecutive time points --> more than 10%
            # decrease in comparison to previous time point would lead to NaN
threshold = 0.1
rows_to_remove = []  # list of rows which should be removed


# if the previous value is NaN, check the value from before the previous value
def check_decreasing_conversion(_value, col_index):
    if np.isnan(_value):
        return False
    previous_col = filtered_columns_conversion[col_index - 1]
    previous_val = df.at[i, previous_col]
    if np.isnan(previous_val):
        if col_index == 1:
            return False
        return check_decreasing_conversion(_value, col_index - 1)
    # if the conversion is decreasing by this much it cannot be used further for the kinetic study and should
    # be thrown out.
    return _value < (previous_val - threshold)


# Iterate through the rows and perform the comparisons
for i in range(1, len(df)):
    for col in range(1, len(filtered_columns_conversion)):
        current_column = filtered_columns_conversion[col]
        previous_column = filtered_columns_conversion[col - 1]
        current_value = df.at[i, current_column]
        previous_value = df.at[i, previous_column]
        # check if both values for comparison are floats, then compare them
        # ToDo: Nachsehen, ob wir hier wirklich ganze Reihe entfernen oder nicht eher Spalten/Zellen @ Yannik: kannst du das bitte überprüfen und dann ggf. die Kommentare verändern?
        if check_decreasing_conversion(current_value, col):
            rows_to_remove.append(i)

# remove rows with conversion below 0
for idx, row in df.iterrows():
    row_values = [row[filtered_col_conv] for filtered_col_conv in filtered_columns_conversion]
    if np.average(row_values) < 0.01:
        if idx not in rows_to_remove:
            rows_to_remove.append(idx)

            # 5.3.4.3. Remove the rows from df and add them to discarded_df
discarded_df3 = df.loc[rows_to_remove].copy()
# Create a new column 'criterion' with a default value in discarded_df
discarded_df3['discarding criterion'] = None
discarded_df3 = discarded_df3.reset_index(drop=True)
discarded_df3.loc[discarded_df3[
                      'discarding criterion'].isna(), 'discarding criterion'] = \
    f'Decreasing conversion within kinetic more than {threshold} from one time point to the next time point'
new_df = pd.concat([discarded_df, discarded_df3])
discarded_df = new_df
# drop all rows from df which should be removed
df = df.drop(rows_to_remove)

# 5.3.4.4. Reset the indices of the dataframes
df = df.reset_index(drop=True)
discarded_df = discarded_df.reset_index(drop=True)

# 5.3.5. remove all datasets (rows) which have less than x (x=4) full (Mn,Mw, D) SEC data points and/or NMR data
# points (conversions)
REMOVER_DECIDER = 4  # number of data points which is necessary to keep the data set
rows_to_remove = []  # list of rows which should be removed

# 5.3.5.1. iterate through the rows
for index, row in df.iterrows():
    is_complete_SEC = 0
    is_complete_NMR = 0

    # 5.3.5.1.1. iterate through the time points for the SEC data points
    # if one of the data points is NaN, add 0 to the is_complete variable
    for time_point in times_list:
        if pd.isna(row[time_point + '-Mn']):
            is_complete_SEC += 0

        elif pd.isna(row[time_point + '-Mw']):
            is_complete_SEC += 0

        elif pd.isna(row[time_point + '-\u00d0']):
            is_complete_SEC += 0
            # if all data points are not NaN, add 1 to the is_complete variable
        else:
            is_complete_SEC += 1

            # 5.3.5.1.2. same as above for NMR data points
    for time_point in times_list:
        # 5.3.5.1.2.1. use all-time points except t6h and t10h (only SEC sampling for those)
        if time_point != 't6h' and time_point != 't10h':
            if pd.isna(row[time_point + '-conversion']):
                is_complete_NMR += 0
            else:
                is_complete_NMR += 1

            # 5.3.5.2 check if the number of complete SEC and NMR data points is smaller than the REMOVER_DECIDER
    if is_complete_NMR < REMOVER_DECIDER or is_complete_SEC < REMOVER_DECIDER:
        rows_to_remove.append(index)

        # 5.3.5.3. Remove the rows from df and add them to discarded_df
discarded_df4 = df.loc[rows_to_remove].copy()
# Create a new column 'criterion' with a default value in discarded_df
discarded_df4['discarding criterion'] = None
discarded_df4 = discarded_df4.reset_index(drop=True)
# Fill the 'discarding criterion' column with new text, where the number of complete data points is smaller than the
# REMOVER_DECIDER
discarded_df4['discarding criterion'] = f'less than {REMOVER_DECIDER} full data points in data set'

new_df = pd.concat([discarded_df, discarded_df4])
discarded_df = new_df
# drop all rows from df which should be removed
df = df.drop(rows_to_remove)

# 5.3.5.4. Reset the indices of the dataframes
df = df.reset_index(drop=True)
discarded_df = discarded_df.reset_index(drop=True)

# 5.3.6. Remove kinetics with decreasing Mn/Mw values
M_SEC_err = 100000 * 2 * 0.10  # the SEC measures up to  100000 g/mol.
# The error for the SEC system of Mn/Mw is 10% so the range is twice
rows_to_remove = []  # list of rows which should be removed

Mw_Mn_get_out_dict = {}
# after the method validation procedure for SEC we throw out all kinetics that are 2x10% lower than after the maximum has been reached
for index, row in df.iterrows():
    # If, over the course of time, the Mn/Mw values fall by more than two times their respective error
    # (% error can be high for the higher point and low for the subsequent, sunken mass value, hence e.g. 2x6=12% distance)
    # the whole kinetic should be discarded
    kinetic_Mw_values = np.array([row[time_point + "-Mw"] for time_point in times_list])
    kinetic_Mw_values = kinetic_Mw_values[~np.isnan(kinetic_Mw_values)]
    highest_M_on_kinetic = kinetic_Mw_values[0]  # set max
    for i in range(1, len(kinetic_Mw_values)):
        # checking if the value is more than M_SEC_p_err lower than the previous one
        curr_point = kinetic_Mw_values[i]
        prev_point = kinetic_Mw_values[i - 1]
        if curr_point > highest_M_on_kinetic:  # do comparison only after a drop
            highest_M_on_kinetic = curr_point

        elif curr_point < highest_M_on_kinetic - M_SEC_err:
            Mw_Mn_get_out_dict[index] = "removed due to Mw sinking more than 2x10% after the maximum has been reached"
            rows_to_remove.append(index)
            break

    kinetic_Mn_values = np.array([row[time_point + "-Mn"] for time_point in times_list])
    kinetic_Mn_values = kinetic_Mn_values[~np.isnan(kinetic_Mn_values)]
    highest_M_on_kinetic = kinetic_Mn_values[0]
    for i in range(1, len(kinetic_Mn_values)):
        curr_point = kinetic_Mn_values[i]
        prev_point = kinetic_Mn_values[i - 1]
        if curr_point > highest_M_on_kinetic:  # do comparison only after a drop
            highest_M_on_kinetic = curr_point

        elif curr_point < highest_M_on_kinetic - M_SEC_err:
            rows_to_remove.append(index)
            if index in Mw_Mn_get_out_dict.keys():
                Mw_Mn_get_out_dict[index] = "removed due to Mw and Mn sinking sinking more than 2x10% after the maximum has been reached"
            else:
                Mw_Mn_get_out_dict[index] = "removed due to Mn sinking more than 2x10% after the maximum has been reached"
                rows_to_remove.append(index)
            break

#  exclude rows from df and add them to discarded_df
discarded_df5 = df.loc[rows_to_remove].copy()
discarded_df5["discarding criterion"] = None
for index in Mw_Mn_get_out_dict.keys():
    discarded_df5.loc[index, "discarding criterion"] = Mw_Mn_get_out_dict[index]
discarded_df5.reset_index(drop=True, inplace=True)

discarded_df = pd.concat([discarded_df, discarded_df5])
discarded_df.reset_index(drop=True, inplace=True)

df.drop(rows_to_remove, inplace=True)
df.reset_index(drop=True, inplace=True)

###################################################################

# 6. check which data is still missing to have performed at least each experiment once
# 6.1. Define the options for the three variables in the experiment
RAFT_agent_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
monomer_options = ['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
solvent_options = ['DMF', 'DMSO', 'Tol']

# 6.2. Create a list to store the permutations
permutations = []

# 6.3. Generate permutations for the options
for RAFT_agent_options, monomer_options, solvent_options in itertools.product(RAFT_agent_options, monomer_options,
                                                                              solvent_options):
    # Build the new string with the chosen options
    new_string = f'{monomer_options}-{RAFT_agent_options}-{solvent_options}'
    permutations.append(new_string)

    # 6.4. Create a dataframe from the list of permutations
permutations_df = pd.DataFrame(permutations, columns=['Missing experiments determiners'])

# 6.5. Compare the permutations dataframe with the curated dataframe from the Excel file
# 6.5.1. Check if the permutations are part of the existing dataframe
#       if so, delete them from the permutations dataframe, which is later printed to the Excel file to see which
#       experiments need to be conducted next
permutations_df.drop(permutations_df[permutations_df['Missing experiments determiners'].isin(
    df['possible sample determiner-original'])].index, inplace=True)

###################################################################


if __name__ == '__main__':
    # 7. save the dataframe to Excel file
    # 7.1. # create an Excel writer object
    with pd.ExcelWriter(OUTPUT_FILE_PATH) as writer:
        # use to_Excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        df.to_excel(writer, sheet_name='utilizable samples', index=False)
        discarded_df.to_excel(writer, sheet_name="discarded samples", index=False)
        #permutations_df.to_excel(writer, sheet_name="Missing experiments", index=False)
