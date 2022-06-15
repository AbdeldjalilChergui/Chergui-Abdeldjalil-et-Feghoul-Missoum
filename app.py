from flask import Flask, request, render_template
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

resultat='0'
mdl='0'

@app.route('/', methods=['GET', 'POST'])
def main():
    global resultat , mdl
    
    if request.method == "POST":
        
        tp = request.form.get("tp")
        model = request.form.get("model")
          
        with open("tf_model.pkl", 'rb') as handle:
            tf1 = pickle.load(handle)
        
        data = tf1.transform([tp])
        dense = data.todense()
        denselist = dense.tolist()
        df1 = pd.DataFrame(denselist)
        
        if model == "Decision tree" :
            clf = joblib.load("dt.pkl")
            mdl = "Decision tree"
        elif model == "Random Forest" :
            clf = joblib.load("RF.pkl")
            mdl = "Random Forest"
        elif model == "SVM" :
            clf = joblib.load("svm.pkl")
            mdl = "SVM"
        elif model == "Logistic Regression" :
            clf = joblib.load("lr.pkl")
            mdl = "Logistic Regression"
        elif model == "MLP classifier" :
            clf = joblib.load("MLP.pkl")
            mdl = "MLP classifier" 
        elif model == "SGD classifier" :      
            clf = joblib.load("sgd.pkl")
            mdl = "SGD classifier"
        
        prediction = clf.predict(df1)        
      
        if prediction==1 :
            resultat='Attaque'
        else :
            resultat='Normal'
            
    else:
        resultat = " "
        mdl = " "
        
    return render_template("xss.html", 
                           output = resultat,
                           model = mdl )

if __name__ == '__main__':
    app.run(debug = True)