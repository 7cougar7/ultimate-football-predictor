import pandas as pd


def main():
    years = ['18', '17', '16', '15', '14', '13']
    for year in years:
        df = pd.read_csv('data/formatted/season20' + year + '.csv')
        df = df[1:]
        df = df.astype({'home score': 'int32', 'visiting score': 'int32'})
        winning_binary_list = [None]
        for row in df.iterrows():
            if row[1][3] > row[1][4]:
                winning_binary_list.append(1)
            elif row[1][3] < row[1][4]:
                winning_binary_list.append(0)
            else:
                winning_binary_list.append(0.5)
        winning_binary = pd.DataFrame(data=winning_binary_list, dtype=float, columns=['Win Binary'])
        winning_binary = winning_binary[1:]
        df = df.join(winning_binary)

        df.to_csv('20' + year + ' Outputs.csv')


if __name__ == '__main__':
    main()