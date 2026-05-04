import pickle

with open("models/svm_emotion_model.pkl", "rb") as f:
    data = pickle.load(f)

# Check the SVM inside the pipeline
pipeline = data['model']
svm = pipeline.steps[-1][1]  # last step is the classifier
print("SVM class_weight:", svm.class_weight)
print("Support vectors per class:", svm.n_support_)
print("Classes (numeric):", svm.classes_)

# Check label encoder to map numbers -> emotion names
le = data['label_encoder']
print("\nLabel mapping (number -> emotion):")
for i, name in enumerate(le.classes_):
    print(f"  {i} = {name}")

# Check training history for class distribution
history = data.get('training_history', {})
print("\nTraining history:", history)