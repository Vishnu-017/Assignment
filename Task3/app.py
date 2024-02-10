
from flask import Flask, render_template,request,jsonify
import pandas as pd

app = Flask(__name__,template_folder='templates')


def get_activity_count():
    df = pd.read_csv('rawdata.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['Date'] = df['date'].dt.date
    data = df.sort_values(by='Date')
    data['Duration'] = data.groupby('location')['Date'].diff()
    data = data.dropna(subset=['Duration'])
    
    return data


@app.route('/')
def activity_count():
    activity_counts = get_activity_count()
    activity_count = activity_counts.groupby(['date', 'activity']).size().unstack(fill_value=0)
    activity_count.columns = ['placing', 'picking']
    html_table =  activity_count.to_html()
    return render_template('activity_count.html', table=html_table)
    

@app.route('/')
def activity_count_with_date():
    data = request.json
    activity_counts = get_activity_count()
    duration_summary = activity_counts.groupby(['date', 'location','position'])['Duration'].sum().reset_index()
    result = duration_summary.to_dict(orient='records')
    return render_template('activity_count_with_date.html', activity_counts=results.to_html())
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
