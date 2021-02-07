import tensorflow as tf
import pandas as pd


def main():
    years = ['20', '19', '18', '17', '16', '15', '14', '13']
    all_inputs = pd.DataFrame()
    for year in years:
        df_team_stats = pd.read_csv('data/results/20' + year + ' Inputs.csv')
        df_known_scores = pd.read_csv('data/results/20' + year + ' Outputs.csv')
        new_df = None
        for idx, row in df_known_scores.iterrows():
            home_team_name = row['home team']
            visiting_team_name = row['visiting team']

            home_stats = df_team_stats.loc[df_team_stats['Team'] == home_team_name]
            home_stats = home_stats.add_prefix('home_')
            home_stats = home_stats.reset_index()
            home_stats = home_stats.drop(['home_Team', 'home_Unnamed: 0', 'index'], axis=1)
            home_stats['home_Year'] = int('20'+year)

            visiting_stats = df_team_stats.loc[df_team_stats['Team'] == visiting_team_name]
            visiting_stats = visiting_stats.add_prefix('visiting_')
            visiting_stats = visiting_stats.reset_index()
            visiting_stats = visiting_stats.drop(['visiting_Team', 'visiting_Unnamed: 0', 'index'], axis=1)
            visiting_stats['visiting_Year'] = int('20'+year)

            total_stats = home_stats.join([visiting_stats])

            df = pd.DataFrame(data=[[row['home score'], row['visiting score'], row['Win Binary']]], dtype='float',
                              columns=['home score', 'visiting score', 'Win Binary'])
            df = df.reset_index()
            df = df.drop(['index'], axis=1)
            # print(df)

            total_stats = total_stats.reset_index()
            total_stats = total_stats.drop(['index'], axis=1)
            total_stats = total_stats.join([df])
            total_stats['Win Binary'] = total_stats['Win Binary'].fillna('*')
            if new_df is None:
                new_df = total_stats
            else:
                new_df = pd.concat([new_df, total_stats])
        all_inputs = pd.concat([all_inputs, new_df])
        new_df.to_csv('data/results/finalized_db20' + year + '.csv')
    all_inputs.to_csv('all_inputs.csv')
    print('Done')


if __name__ == '__main__':
    main()
