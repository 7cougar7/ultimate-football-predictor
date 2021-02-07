import csv


def main():
    years = [20, 19, 18, 17, 16, 15, 14, 13]
    for year in years:
        with open('data/unformatted/nfl odds 20' + str(year) + '-' + str(year+1) + '.xlsx - Sheet1.csv') as fin:
            reader = csv.reader(fin)
            new_row = ["", "", "", "", ""]
            file_contents = [["date", 'home team', 'visiting team', 'home score', 'visiting score']]
            for row_num, row in enumerate(reader):
                if row_num == 0:
                    continue
                if row_num % 2 == 1:
                    print(new_row)
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
                if row[2] == 'H':
                    new_row[1] = row[3]
                    new_row[3] = row[8]
                elif row[2] == 'V':
                    new_row[2] = row[3]
                    new_row[4] = row[8]
                else:
                    new_row[row_num%2+1] = row[3]
                    new_row[row_num%2+3] = row[8]
            with open('season20'+str(year)+'.csv', 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(file_contents)


if __name__ == '__main__':
    main()
