from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('Titanic_Model.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template("dropdown.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        pclass = request.form.get("pclass", None)
        gender = request.form.get("gender", None)
        age = request.form.get('age')
        fare = request.form.get('fare')
        all = np.array([pclass,gender,age,fare])
        pre = model.predict([all])
        pred = np.asscalar(pre)  #0 dead , 1 survived
    return render_template("dropdown.html", all=pred)

if __name__ == '__main__':
    app.run(debug=True)