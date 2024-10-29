from flask import Flask, render_template, request
import pickle
import numpy as np

# setup application
app = Flask(__name__,template_folder='template')

def prediction(lst):
    filename = 'model/rf_model_sel.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([lst])
    return pred_value

@app.route('/', methods=['POST', 'GET'])
def index():
    # return "Hello World"
    pred_value = 0
    if request.method == 'POST':
        AGE = request.form['AGE']
        SALARY = request.form['SALARY']
        N_OF_DEPENDENTS = request.form['N_OF_DEPENDENTS']
        MORTGAGE_AMOUNT = request.form['MORTGAGE_AMOUNT']
        TIME_AS_CUSTOMER = request.form['TIME_AS_CUSTOMER']
        N_MORTGAGES = request.form['N_MORTGAGES']
        HAS_CHILDREN = request.form['HAS_CHILDREN']
        HOUSE_OWNERSHIP = request.form['HOUSE_OWNERSHIP']
        
        feature_list = []
        HAS_CHILDREN_list=['1']
        HOUSE_OWNERSHIP_list = ['1','2']
        def traverse_list(lst, value):
            for item in lst:
                if item == value:
                    feature_list.append(1)
                else:
                    feature_list.append(0)
        

        traverse_list(HAS_CHILDREN_list, HAS_CHILDREN)
        traverse_list(HOUSE_OWNERSHIP_list, HOUSE_OWNERSHIP)
        feature_list.append(float(MORTGAGE_AMOUNT))
        feature_list.append(int(AGE))
        feature_list.append(int(N_OF_DEPENDENTS))
        feature_list.append(int(N_MORTGAGES))
        feature_list.append(float(SALARY))
        feature_list.append(int(TIME_AS_CUSTOMER))
        
        print(feature_list)
        pred_value = prediction(feature_list)
        print(pred_value)
        pred_value = np.round(pred_value[0],3)
        pred_value
    return render_template('index.html', pred_value=pred_value)


if __name__ == '__main__':
    app.run(debug=True)