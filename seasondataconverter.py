import csv


def main():
    year_value = 2013
    with open('data/unformatted/nfl odds 2013-14.xlsx - Sheet1.csv') as fin:
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
                    year = str(year_value + 1)
                else:
                    year = str(year_value)
                new_row[0] = month + '-' + day + '-' + year
            if row[2] == 'H':
                new_row[1] = row[3]
                new_row[3] = row[8]
            else:
                new_row[2] = row[3]
                new_row[4] = row[8]
        with open('season'+str(year_value)+'.csv', 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(file_contents)


if __name__ == '__main__':
    main()
