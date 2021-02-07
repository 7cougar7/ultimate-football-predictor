import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    name_dict = {
        'Arizona Cardinals': 'Arizona',
        'Arizona': 'Arizona',
        'Atlanta Falcons': 'Atlanta',
        'Atlanta': 'Atlanta',
        'Baltimore Ravens': 'Baltimore',
        'Baltimore': 'Baltimore',
        'Buffalo Bills': 'Buffalo',
        'BuffaloBills': 'Buffalo',
        'Buffalo': 'Buffalo',
        'Carolina Panthers': 'Carolina',
        'Carolina ': 'Carolina',
        'Carolina': 'Carolina',
        'Chicago Bears': 'Chicago',
        'Chicago': 'Chicago',
        'Cincinnati Bengals': 'Cincinnati',
        'Cincinnati': 'Cincinnati',
        'Cleveland Browns': 'Cleveland',
        'Cleveland': 'Cleveland',
        'Dallas Cowboys': 'Dallas',
        'Dallas': 'Dallas',
        'Denver Broncos': 'Denver',
        'Denver': 'Denver',
        'Detroit Lions': 'Detroit',
        'Detroit': 'Detroit',
        'Green Bay Packers': 'GreenBay',
        'GreenBay': 'GreenBay',
        'Green Bay': 'GreenBay',
        'Houston Texans': 'Houston',
        'Houston': 'Houston',
        '  Houston': 'Houston',
        'Indianapolis Colts': 'Indianapolis',
        'Indianapolis': 'Indianapolis',
        'Jacksonville Jaguars': 'Jacksonville',
        'Jacksonville': 'Jacksonville',
        'Kansas City Chiefs': 'KansasCity',
        'KansasCity': 'KansasCity',
        'Kansas City': 'KansasCity',
        'KCChiefs': 'KansasCity',
        'Kansas': 'KansasCity',
        'Miami Dolphins': 'Miami',
        'Miami': 'Miami',
        'Minnesota Vikings': 'Minnesota',
        'Minnesota': 'Minnesota',
        ' Minnesota': 'Minnesota',
        'New England Patriots': 'NewEngland',
        'NewEngland': 'NewEngland',
        'New England': 'NewEngland',
        'New Orleans Saints': 'NewOrleans',
        'NewOrleans': 'NewOrleans',
        'New Orleans': 'NewOrleans',
        'New York Giants': 'NYGiants',
        'NYGiants': 'NYGiants',
        'NY Giants': 'NYGiants',
        'NewYork': 'NYGiants',
        'New York Jets': 'NYJets',
        'NYJets': 'NYJets',
        'NY Jets': 'NYJets',
        'Las Vegas Raiders': 'LasVegas',
        'Las Vegas': 'LasVegas',
        'Oakland Raiders': 'LasVegas',
        'Oakland': 'LasVegas',
        'LVRaiders': 'LasVegas',
        'LasVegas': 'LasVegas',
        'Philadelphia Eagles': 'Philadelphia',
        'Philadelphia': 'Philadelphia',
        'Pittsburgh Steelers': 'Pittsburgh',
        'Pittsburgh': 'Pittsburgh',
        'San Diego Chargers': 'LAChargers',
        'Los Angeles Chargers': 'LAChargers',
        'LAChargers': 'LAChargers',
        'LA Chargers': 'LAChargers',
        'SanDiego': 'LAChargers',
        'San Francisco 49ers': 'SanFrancisco',
        'SanFrancisco': 'SanFrancisco',
        'San Francisco': 'SanFrancisco',
        'Seattle Seahawks': 'Seattle',
        'Seattle': 'Seattle',
        'St. Louis Rams': 'LARams',
        'St.Louis': 'LARams',
        'Los Angeles Rams': 'LARams',
        'LosAngeles': 'LARams',
        'LARams': 'LARams',
        'LA Rams': 'LARams',
        'Tampa Bay Buccaneers': 'TampaBay',
        'Tampa': 'TampaBay',
        'TampaBay': 'TampaBay',
        'Tampa Bay': 'TampaBay',
        'Tennessee Titans': 'Tennessee',
        'Tennessee': 'Tennessee',
        'Washington Redskins': 'Washington',
        'Washington Football Team': 'Washington',
        'Washington': 'Washington',
        'Washingtom': 'Washington',
        'ARI': 'Arizona',
        'ATL': 'Atlanta',
        'BAL': 'Baltimore',
        'BUF': 'Buffalo',
        'CAR': 'Carolina',
        'CHI': 'Chicago',
        'CIN': 'Cincinnati',
        'CLE': 'Cleveland',
        'DAL': 'Dallas',
        'DEN': 'Denver',
        'DET': 'Detroit',
        'GNB': 'GreenBay',
        'HOU': 'Houston',
        'IND': 'Indianapolis',
        'JAX': 'Jacksonville',
        'KAN': 'KansasCity',
        'LAC': 'LAChargers',
        'LAR': 'LARaiders',
        'OAK': 'LARaiders',
        'LAS': 'LasVegas',
        'MIA': 'Miami',
        'MIN': 'Minnesota',
        'NE': 'NewEngland',
        'NOR': 'NewOrleans',
        'NO': 'NewOrleans',
        'NYG': 'NYGiants',
        'NYJ': 'NYJets',
        'PHI': 'Philadelphia',
        'PIT': 'Pittsburgh',
        'SAN': 'SanFrancisco',
        'SEA': 'Seattle',
        'TAM': 'TampaBay',
        'TEN': 'Tennessee',
        'WAS': 'Washington',
    }
    years = ['20', '19', '18', '17', '16', '15', '14', '13']
    # years = ['20']
    for year in years:
        required_cols_off = [1, 7, 12, 18]
        df = pd.read_csv('OffensiveRating20' + year + '.csv', usecols=required_cols_off)
        df = df[1:]
        df.columns = ['Team', 'Off Turnovers', 'Passing Yards', 'Rushing Yards']
        df = df.astype({'Off Turnovers': 'int32', 'Passing Yards': 'int32', 'Rushing Yards': 'int32'})
        df['Team'] = df['Team'].map(name_dict).fillna(df['Team'] + '~~~')
        names = df[['Team']]
        df.sort_values('Team', inplace=True)
        df = df.reset_index()
        df = df.drop(['index'], axis=1)
        # df = df.drop('Team', axis=1)
        # df = names.join([df])

        required_cols_def = [1, 7, 12, 18]
        df_def = pd.read_csv('DefensiveRating20' + year + '.csv', usecols=required_cols_def)
        df_def = df_def[1:]
        df_def.columns = ['Team', 'Def Turnovers', 'Passing Yards All', 'Rushing Yards All']
        df_def = df_def.astype({'Def Turnovers': 'int32', 'Passing Yards All': 'int32', 'Rushing Yards All': 'int32'})
        df_def['Team'] = df_def['Team'].map(name_dict).fillna(df_def['Team'] + '~~~')
        names = df_def[['Team']]
        df_def.sort_values('Team', inplace=True)
        df_def = df_def.reset_index()
        df_def = df_def.drop(['index', 'Team'], axis=1)
        # df_def = df_def.drop('Team', axis=1)
        # df_def = names.join([df_def])
        # df.sort_values('Team', inplace=True)
        # df_def = df_def.drop('Team', axis=1)
        df_total = df.join([df_def])

        required_cols_off = [1, 2]
        df_qbr = pd.read_csv('QBR20' + year + '.csv', usecols=required_cols_off)
        # df = df[1:]
        df_qbr.columns = ['Tm', 'QBR']
        df_qbr = df_qbr.astype({'QBR': 'float'})
        df_qbr['Tm'] = df_qbr['Tm'].map(name_dict).fillna(df_qbr['Tm'] + '~~~')
        df_qbr['QBR'] = df_qbr['QBR'].fillna('~~~')
        df_qbr.sort_values('Tm', inplace=True)
        df_qbr = df_qbr.reset_index()
        df_qbr = df_qbr.drop(['index', 'Tm'], axis=1)
        df_total = df_total.join([df_qbr])

        # The GOAT CODE
        df_total.to_csv('data/results/20' + year + ' Inputs.csv')
    print('Done')

        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # print(df_total)
        # with pd.option_context('display.max_seq_items', None):
        #    print(df.columns)

    # required_cols_off = [1, 7, 12, 18]
    # df = pd.read_csv('OffensiveRatings2020.csv', usecols=required_cols_off)
    # df = df[1:]
    # df.columns = ['Team', 'Off Turnovers', 'Passing Yards', 'Rushing Yards']
    # df = df.astype({'Off Turnovers': 'int32', 'Passing Yards': 'int32', 'Rushing Yards': 'int32'})
    # # totals = df.sum(axis=0, numeric_only=True)
    # # totals = totals / 32
    # names = df[['Team']]
    # df = df.drop('Team', axis=1)
    # #x = df.values()
    # #min_max_scalar = preprocessing
    # # df = df.divide([totals['Off Turnovers'], totals['Passing Yards'], totals['Rushing Yards']])
    # df = names.join([df])
    #
    #
    # required_cols_def = [1, 7, 12, 18]
    # df_def = pd.read_csv('Defensive Stats.csv', usecols=required_cols_def)
    # df_def = df_def[1:]
    # df_def.columns = ['Team', 'Def Turnovers', 'Passing Yards All', 'Rushing Yards All']
    # df_def = df_def.astype({'Def Turnovers': 'int32', 'Passing Yards All': 'int32', 'Rushing Yards All': 'int32'})
    # # totals_def = df_def.sum(axis=0, numeric_only=True)
    # # totals_def = totals_def / 32
    # names = df_def[['Team']]
    # df_def = df_def.drop('Team', axis=1)
    # # df_def = df_def.divide(
    # #     [totals_def['Def Turnovers'], totals_def['Passing Yards All'], totals_def['Rushing Yards All']])
    # df_def = names.join([df_def])
    # df.sort_values('Team', inplace=True)
    # df_def.sort_values('Team', inplace=True)
    # df_def = df_def.drop('Team', axis=1)
    # df_total = df.join([df_def])
    #
    # required_cols_off = [2]
    # df = pd.read_csv('QB Ratings2020.csv', usecols=required_cols_off)
    # df = df[1:]
    # df.columns = ['QBR']
    # df = df.astype({'QBR': 'float'})
    # totals = df.sum(axis=0, numeric_only=True)
    # # totals = totals / 32
    # # df = df.divide([totals['QBR']])
    # df_total = df_total.join([df])
    #
    # # The GOAT CODE
    # df_total.to_csv('2020 Inputs.csv')
    #
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(df_total)
    # # with pd.option_context('display.max_seq_items', None):
    # #    print(df.columns)


if __name__ == '__main__':
    main()
    # pass
