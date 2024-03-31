from flask import Flask, request, render_template
from glob import glob
import mne
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open("op.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = f"uploaded_files/{file.filename}"
            file.save(file_path)
            
            # Perform prediction
            prediction = predict(file_path)
            
            return render_template('result.html', prediction=prediction)

    return render_template('index.html')

def predict(file_path):
    # Read and preprocess the data
    datax = mne.io.read_raw_edf(file_path, preload=True)
    datax.set_eeg_reference()
    datax.filter(l_freq=1, h_freq=45)
    epochs = mne.make_fixed_length_epochs(datax, duration=25, overlap=0)
    epochs = epochs.get_data()
    
    # Extract features
    from scipy import stats
    def mean(data):
        return np.mean(data,axis=-1)
        
    def std(data):
        return np.std(data,axis=-1)

    def ptp(data):
        return np.ptp(data,axis=-1)

    def var(data):
            return np.var(data,axis=-1)

    def minim(data):
        return np.min(data,axis=-1)


    def maxim(data):
        return np.max(data,axis=-1)

    def argminim(data):
        return np.argmin(data,axis=-1)


    def argmaxim(data):
        return np.argmax(data,axis=-1)

    def mean_square(data):
        return np.mean(data**2,axis=-1)

    def rms(data): #root mean square
        return  np.sqrt(np.mean(data**2,axis=-1))  

    def abs_diffs_signal(data):
        return np.sum(np.abs(np.diff(data,axis=-1)),axis=-1)


    def skewness(data):
        return stats.skew(data,axis=-1)

    def kurtosis(data):
        return stats.kurtosis(data,axis=-1)

    def concatenate_features(data):
        return np.concatenate((mean(data),std(data),ptp(data),var(data),minim(data),maxim(data),argminim(data),argmaxim(data),
                            mean_square(data),rms(data),abs_diffs_signal(data),
                            skewness(data),kurtosis(data)),axis=-1)
    
    features = [concatenate_features(data) for data in epochs]
    features = np.array(features)
    
    # Perform prediction
    prediction = clf.predict(features)
    return prediction[0]

if __name__ == '__main__':
    app.run(debug=True)
