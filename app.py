from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)


# app.config['CORS_HEADERS'] = 'Content-Type'

# A welcome message to test our server
@app.route('/')
def index():
    return render_template('homepage.html')


@app.route('/get_team_info/', methods=['POST'])
def get_team_info():
    name = request.form.get('name')
    year = request.form.get('year')
    if name and year:
        df = pd.read_csv('static/input_data/' + year + 'Inputs.csv')
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        names = df[['Team']]
        df.drop(['Team'], axis=1, inplace=True)
        df = (df - df.min()) / (df.max() - df.min())
        row = df.loc[names['Team'] == name]
        # row.drop(['Team'], axis=1, inplace=True)
        row = row.to_numpy().tolist()[0]
        row.append((int(year) - 2013) / 7)
        return jsonify({'team': row})
    else:
        return jsonify({})


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
