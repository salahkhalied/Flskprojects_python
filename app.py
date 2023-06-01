from flask import Flask, render_template, url_for ,request
import pandas as pd
from tabulate  import tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import  matplotlib.pyplot as plt
import os
import  seaborn as sns
import pickle
app = Flask(__name__ )
data = pd.read_csv("heartnew.csv").head()
#print(data.columns)
data1 = pd.read_csv("heartnew.csv")
pd1 = pd.DataFrame(data)
desc = data1.describe()
x = data1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values
y = data1.iloc[:, 13].values
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25 , random_state=0)
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
#xtest = sc_x.transform(xtest)
scaler_file = 'scalerNN.sav'
sc_x = pickle.load(open(scaler_file, 'rb'))

#scaler_file = 'scaler_SVM.sav'
sc_x_svm = pickle.load(open('scaler_SVM.sav', 'rb'))

#xtest = sc_x.transform(xtest)
xlenth=0
#print (xtrain[0:10, :])
rex = tabulate(pd1 ,headers='keys', tablefmt="html" , showindex="always"  )
rex1 = tabulate(desc ,headers='keys',tablefmt="html" , showindex="always")
rex2 = tabulate(xtrain[0:10, :] , headers='keys',tablefmt='html',showindex='always')
@app.route("/") # like index
def home():
    return render_template('index.html' )
@app.route("/Busness")
def Busness():
    target = data1['output'].value_counts()
    sex = data1['sex'].value_counts()
    return render_template('Describe.html', tables= [rex], titles=rex.title(),tables1=[rex1]
                           ,titles1=rex1.title(), ctitles1=rex1.title() ,
                           output =target , sexoutput = sex )

@app.route("/DataUnderstand")
def DataUnderstand():
    return render_template('Explain.html')

@app.route("/Preprocssing")
def Preprocssing():
    return render_template('Preprocssing.html' , tables= [rex2], titles=[rex2.title()])

#==========================================================
@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    output=''
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        xlenth=len(int_features)

        # print("intial values -->", int_features)
        pre_final_features = [np.array(int_features)]
        final_features = sc_x.transform(pre_final_features)
        #model = pickle.load(open('NerrulNetworkClassifier.pkl', 'rb'))
        # scaler = pickle.load(open('scalerNerulNetwork.pickle', 'rb'))
        # ---------------------------------------------
        # load scaler

        # -------------------------------------
        # load the model from disk
        mpdel_file = 'model_NN.sav'
        classifier = pickle.load(open("model_NN.sav", 'rb'))
        # ------------------------------------
       # y_pred = classifier.predict(xtest)
        #print("Accuracy : ", accuracy_score(ytest, y_pred))
        # ---------------------------------------------

        #print("scaled values -->", final_features)
        prediction = classifier.predict(final_features)
        print('predictio value is ', prediction[0])
        if (prediction[0] == 1):
            output = "True"
        elif (prediction[0] == 0):
                output = "False"
        else:
         output = "Not sure"
    return render_template('ModelNureal.html', prediction_text= format(output))
#========================================Predict SVM========================================================
@app.route('/predictSVM', methods=['GET', 'POST'])
def predictSVM():
    output=''
    if request.method == 'POST':

        int_features = [int(x) for x in request.form.values()]
        pre_final_features = [np.array(int_features)]
        final_features = sc_x_svm.transform(pre_final_features)
        classifier_svm = pickle.load(open("model_SVM.sav", 'rb'))
        prediction = classifier_svm.predict(final_features)
        print('predictio value is ', prediction[0])
        if (prediction[0] == 1):
            output = "True"
        elif (prediction[0] == 0):
                output = "False"
        else:
         output = "Not sure"
    return render_template('ModelSVM.html', prediction_text= format(output))
#============================================================================================================
if __name__ == '__main__':
    app.run(host='localhost', port=5555, debug=True)
    #app.run()
