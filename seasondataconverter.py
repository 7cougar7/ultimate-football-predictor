import csv


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
    years = [20, 19, 18, 17, 16, 15, 14, 13]
    for year in years:
        with open('data/unformatted/nfl odds 20' + str(year) + '-' + str(year + 1) + '.xlsx - Sheet1.csv') as fin:
            reader = csv.reader(fin)
            new_row = ["", "", "", "", ""]
            file_contents = [["date", 'home team', 'visiting team', 'home score', 'visiting score']]
            for row_num, row in enumerate(reader):
                if row_num == 0:
                    continue
                if row_num % 2 == 1:
                    # print(new_row)
                    file_contents.append(new_row)
                    new_row = ["", "", "", "", ""]
                if new_row[0] == "":
                    date = row[0]
                    day = date[-2:]
                    month = date[0:(len(date) - 2)]
                    if int(month) < 6:
                        year_str = str(year + 1)
                    else:
                        year_str = str(year)
                    new_row[0] = month + '-' + day + '-' + year_str
                if row[3] not in name_dict.keys():
                    print('ERROR: Could not find value |' + row[3] + '| in name dictionary. Year:'+str(year))
                    return -1
                if row[2] == 'H':
                    new_row[1] = name_dict[row[3]]
                    new_row[3] = row[8]
                elif row[2] == 'V':
                    new_row[2] = name_dict[row[3]]
                    new_row[4] = row[8]
                else:
                    new_row[row_num % 2 + 1] = name_dict[row[3]]
                    new_row[row_num % 2 + 3] = row[8]
            with open('data/formatted/season20' + str(year) + '.csv', 'w', newline='') as fout:
                writer = csv.writer(fout)
                file_contents.pop(1)
                writer.writerows(file_contents)
    print('Done')


if __name__ == '__main__':
    main()
