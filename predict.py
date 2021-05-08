from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import math

#model = load_model('my_model')
#model = load_model('saved_2hrs_unfitted_model')
model = load_model('superimposed_model')
#model = load_model('my_model_python_img')


#dir_path = 'Unfitted/testing/Up'
dir_path = 'spy_2hrs/data/order_408684/SPY_Plots/Fitted_2_hours/testing/Down'
dir_path = 'two_hrs/data/order_408684/SPY_Plots/Fitted_11/testing/Down'
dir_path = '2hrs_python_generated/data/Aws_Plots/Python/testing/Down'
zeros = 0
total = 0
for i in os.listdir(dir_path):
    filename = '{}//{}'.format(dir_path, i)
    print('filename: {}'.format(filename))

    img = image.load_img(filename, target_size=(800, 800))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    print('prediction: {}'.format(val))
    pred = model.predict_proba(images)
    print('confidence: {}\n\n'.format(pred))
    if round(pred[0][0]) == 0:
        zeros += 1

    total += 1


pct = ((1.0*zeros)/total) * 100
print('pct {}%'.format(pct))
print('{}/{}'.format(zeros, total))

