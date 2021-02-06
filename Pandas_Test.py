import pandas as pd
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    required_cols_off = [1, 7, 12, 18]
    df = pd.read_csv('OffensiveRatings2020.csv', usecols=required_cols_off)
    df = df[1:]
    df.columns = ['Team', 'Off Turnovers', 'Passing Yards', 'Rushing Yards']
    df = df.astype({'Off Turnovers': 'int32', 'Passing Yards': 'int32', 'Rushing Yards': 'int32'})
    totals = df.sum(axis=0, numeric_only=True)
    #totals = totals.divide([1, 16, 16], axis=0)
    totals = totals/32
    #df = df[['Turnovers']]/totals[['Turnovers']]
    names = df[['Team']]
    df = df.drop('Team', axis=1)
    df = df.divide([totals['Off Turnovers'], totals['Passing Yards'], totals['Rushing Yards']])
    df = names.join([df])

    required_cols_def = [1, 7, 12, 18]
    df_def = pd.read_csv('Defensive Stats.csv', usecols=required_cols_def)
    df_def = df_def[1:]
    df_def.columns = ['Team', 'Def Turnovers', 'Passing Yards All', 'Rushing Yards All']
    df_def = df_def.astype({'Def Turnovers': 'int32', 'Passing Yards All': 'int32', 'Rushing Yards All': 'int32'})
    totals_def = df_def.sum(axis=0, numeric_only=True)
    totals_def = totals_def/32
    names = df_def[['Team']]
    df_def = df_def.drop('Team', axis=1)
    df_def = df_def.divide([totals_def['Def Turnovers'], totals_def['Passing Yards All'], totals_def['Rushing Yards All']])
    df_def = names.join([df_def])
    df.sort_values('Team', inplace=True)
    df_def.sort_values('Team', inplace=True)
    df_def = df_def.drop('Team', axis=1)
    #df = pd.concat([df, df_def], axis=1, sort=True)
    df = df.join([df_def])




    
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
    #with pd.option_context('display.max_seq_items', None):
    #    print(df.columns)


if __name__ == '__main__':
    main()
