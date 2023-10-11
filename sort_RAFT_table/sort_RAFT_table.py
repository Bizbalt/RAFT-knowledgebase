INPUT_FILE_PATH = "C:\\Users\\xo37lay\\Desktop\\2023_07_07 - evaluation table (NMR and SEC).xlsx"
OUTPUT_FILE_PATH = "C:\\Users\\xo37lay\\Desktop\\2023_07_07 - evaluation table (NMR and SEC)_temp.xlsx"

from this import d
import pandas as pd
import os

#1. read data from excel file to pd dataframe
df = pd.read_excel(INPUT_FILE_PATH) #reads excel file to pandas dataframe


#2. restructure the dataframe
    
df.dropna(inplace = True, how = "all") #drop all completely empty rows
df = df.drop(df.columns[0],axis=1) #drop first column (legend in the excel file)
df.reset_index(drop=True, inplace=True) #reset index after row removing

    #2.1. change header of dataframe to more readable names (recently in header only t0h, t1h, t2h, etc. or unnamed)-->(e.g., t0h-Mn)
        #2.1.1. change header of first column(Legend) to "Unnamed 1"
column_names = df.columns
df.rename(columns={column_names[0]: "Unnamed 1"}, inplace=True)
        #2.1.2. replace header (first row in original document contained only information about time of sampling) with the second row (information about the data in the column (without time of sampling)))
new_header = df.iloc[0] #grab the first row for the header
df = df[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header

        #2.1.3. add information about time to the header (e.g. t0h-Mn)
for col in df.columns:
    if not any(keyword in col for keyword in ['sample determiner', 'reactor', 'date', 'solution', 'data', 'comments']):
        # If non of the keywords is found in the column name, change the column name
            
        
        
            #erstes Kriterium: t0h, t1h, t2h, etc. muss in den Namen rein (startend bei t0h). Wenn erkannt wird, dass es den dann kommenden header (z.B. Mn) schon gibt, 
            ##dann wird der folgende header mit der neuen Zeit (z.B. t1h) versehen
            #zweites Kriterium: die zweite Zeile wird zur Kommentarspalte für den header entweder "(" als Startpunkt des Kommentars oder "-->"
            #drittes Kriterium: dritte Zeile wird zur Einheitenzeile "[" als Startpunkt dafür 
        new_col_name = col + 'iii'
        df.rename(columns={col: new_col_name}, inplace=True)






#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'Overview of data in data frame'

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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#print(df.columns) #show all column names
df.to_excel(OUTPUT_FILE_PATH, index=False)