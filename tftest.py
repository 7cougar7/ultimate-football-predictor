import tensorflow as tf
import pandas as pd


def main():
    df_team_stats = pd.read_csv('2020 Inputs.csv')
    df_known_scores = pd.read_csv('2020 Outputs.csv')
    new_df = None
    for idx, row in df_known_scores.iterrows():
        home_team_name = row['home_team']
        visiting_team_name = row['visiting_team']

        home_stats = df_team_stats.loc[df_team_stats['team'] == home_team_name]
        home_stats = home_stats.add_prefix('home_')
        home_stats = home_stats.reset_index()
        home_stats = home_stats.drop(['home_team', 'home_Unnamed: 0', 'index'], axis=1)

        visiting_stats = df_team_stats.loc[df_team_stats['team'] == visiting_team_name]
        visiting_stats = visiting_stats.add_prefix('visiting_')
        visiting_stats = visiting_stats.reset_index()
        visiting_stats = visiting_stats.drop(['visiting_team', 'visiting_Unnamed: 0', 'index'], axis=1)

        total_stats = home_stats.join([visiting_stats])

        df = pd.DataFrame(data=[[row['home_score'], row['visiting_score'], row['win_binary']]], dtype='float',
                          columns=['home_score', 'visiting_score', 'win_binary'])
        df.drop(df.tail(0).index, inplace=True)
        total_stats = total_stats.join([df])
        if new_df is None:
            new_df = total_stats
        else:
            new_df = pd.concat([new_df, total_stats])
    new_df.to_csv('finalized_db.csv')


if __name__ == '__main__':
    main()
