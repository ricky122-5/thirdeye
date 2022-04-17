from ctypes import resize
from django.shortcuts import render
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
import os
import torch
print(tf.__version__)

# code for loading model: keras.models.load_model(PATH) --> returns model object
PATH = os.getcwd()+'\diseasedetector\cnn_model2'
model = keras.models.load_model(PATH)

PATH1 = os.getcwd()+'\diseasedetector\\best.torchscript.pt'
model1 = torch.load(PATH1)
model1.eval()
# code for making prediction with model: model.predict(np.array([x])) (x is a single example)

# code for converting prediction probability to binary value: preds >= 0.5
# 1 --> covid, 0 --> healthy

# code for opening image as numpy array:
# path = ...
# arr = np.asarray(Image.open(path))/255.

# Create your views here

def get_probs_and_labels(results):
  probs_softmax = tf.nn.softmax(results, axis=-1).numpy()
  bins = np.argmax(probs_softmax, axis=-1)
  probs = []
  labels = []
  for i in range(len(bins)):
    if bins[i] == 1:
      labels.append("COVID-19")
      probs.append(probs_softmax[i][1])
    else:
      labels.append("HEALTHY")
      probs.append(probs_softmax[i][0])

  return probs, labels

def home(request):
    if (request.method=="POST"):
        global fileString1
        theFile = request.FILES["text"]
        fs = FileSystemStorage()
        filename = fs.save(theFile.name, theFile)
        uploaded_file_url = fs.url(filename)
        fileString1 = str(uploaded_file_url)
        print(fileString1)
        return HttpResponseRedirect('predict')
    return render(request, 'home.html')

def predict(request):
    resizedImg = (Image.open('C:/Users/Rikki/Downloads/HackTJ/HackTJ'+fileString1)).resize((75,75))
    arr = np.asarray(resizedImg)/255.
    arr = np.stack([arr,arr,arr],axis=(-1))
    result = model.predict(np.array([arr]))
    probs,labels = get_probs_and_labels(result)
    prob = probs[0]
    label = labels[0]
    arr1 = np.asarray((Image.open('C:/Users/Rikki/Downloads/HackTJ/HackTJ'+fileString1).resize((512,512))))
    arr1 = np.stack([arr1,arr1,arr1],axis=(0))
    arr1 = np.array([arr1])
    torcharr = torch.from_numpy(arr1).float().detach()

    result1 = model1(torcharr)[0].numpy()
    context = {
        'probability': prob,
        'label': label,
        'numpy': result1,
    }
    return render(request, 'predict.html', context)



# Initialize Firebase

# Signup












