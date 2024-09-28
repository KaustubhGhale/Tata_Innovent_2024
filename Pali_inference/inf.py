import inference #Requires python version from 3.8-3.11
model = inference.get_model("full-dataset-7rmc3/1")
model.infer(image="/home/prakhar/Desktop/Tata Competition/Car Damage Detection.v4i.paligemma/dataset/0001_JPEG.rf.8cce9bb7a46ff7494b475d1ab652324a.jpg")