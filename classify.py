from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

model_name = "./model/mymodel_echocv_500-500-8_adam_16_0.9394.h5"
input_shape = (224,224,3)
batch_size = 8

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    directory='./sample/',
    target_size=input_shape[:2],
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle=False
)
STEP_SIZE_TEST = test_generator.n/batch_size
test_generator.reset()

model = load_model(model_name)
pred = model.predict_generator(test_generator, verbose=1)

confidence = ["{0:.3f}".format(np.amax(p)) for p in pred]
predicted_class_indices = np.argmax(pred,axis=1)
labels = { 'plax':0, 'psax-av':1, 'psax-mv':2, 'psax-ap':3, 'a4c':4, 'a5c':5, 'a3c':6, 'a2c':7 }
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results = pd.DataFrame({"Filename":filenames, "Prediction":predictions, "Confidence":confidence})
results_file = "./results.csv"
results.to_csv(results_file,index=False)