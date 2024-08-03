import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import sklearn.metrics
from sklearn.metrics import *
from sklearn.metrics._dist_metrics import *
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import pickle
import json
import traceback
import logging



app = Flask('Fetal health',template_folder='templates')
# app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config.from_object(__name__)

log = logging.getLogger('Fetal health')

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# model =joblib.load("fetal_health_knn.pkl","r+")
model = pickle.load(open("fetal_health_random_forest.pkl", "rb"))
# clf1 = pickle.load(open('fetal_health_knn.pkl', 'rb'))
# print(traceback.format_exc())
# model=joblib.load("fetal_health_knn.pkl")
# model=pickle.load(open("fetal","rb"))


df = pd.DataFrame()


@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    input_features = { k:float(v) for k ,v in request.form.to_dict().items()}
    
    # log.debug(input_features)
    
    if not (100 <= input_features.get('baseline_value') <= 180 and
            0 <= input_features.get('prolongued_decelerations') <= 0.005 and
            50 <= input_features.get('histogram_mode') <= 200 and
            0 <= input_features.get('mean_value_of_long_term_variability') <= 200 and
            70 <= input_features.get('histogram_median') <= 190 and
            70 <= input_features.get('histogram_mean') <= 190 and
            0 <= input_features.get('accelerations') <= 0.05 and
            0 <= input_features.get('mean_value_of_short_term_variability') <= 10 and
            0 <= input_features.get('percentage_of_time_with_abnormal_long_term_variability') <= 100 and
            10 <= input_features.get('abnormal_short_term_variability') <= 100):
        
        return render_template('predict.html', Error='There is invaild values in response! please check it again')
    
    # print(traceback.format_exc())
    
    # log.debug('Processing request for index route')

    output = model.predict(np.array(tuple(input_features.values())).reshape(1, -1))[0]
    
    clasification  =  'Normal' if output == 1 else 'Suspect' if output == 2 else 'Pathological' 
    
    print("Predicted output:", output) 

    return render_template('predict.html', prediction_text=clasification)



if __name__ == "__main__":
    # print(traceback.format_exc())
    app.run(debug=True)
    # print(traceback.format_exc())