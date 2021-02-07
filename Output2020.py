import pandas as pd


def main():
    years = ['20', '19', '18', '17', '16', '15', '14', '13']
    for year in years:
        df = pd.read_csv('data/formatted/season20' + year + '.csv')
        # df = df[1:]
        df = df.astype({'home score': 'int32', 'visiting score': 'int32'})
        winning_binary_list = []
        for idx, row in df.iterrows():
            if row[3] > row[4]:
                winning_binary_list.append(1)
            elif row[3] < row[4]:
                winning_binary_list.append(0)
            else:
                winning_binary_list.append(0.5)
        winning_binary = pd.DataFrame(data=winning_binary_list, dtype=float, columns=['Win Binary'])
        # winning_binary.drop(winning_binary.head(0).index, inplace=True)
        print(winning_binary)
        # winning_binary = winning_binary.reset_index()
        # winning_binary = winning_binary.drop(['index'], axis=1)
        df = df.join(winning_binary)

        df.to_csv('data/results/20' + year + ' Outputs.csv')
    print('Done')

if __name__ == '__main__':
    main()
