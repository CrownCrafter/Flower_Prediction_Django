from django.shortcuts import render
from django.http import JsonResponse
# Each page in a function(requrest as param)
def home(request):
    return render(request, 'index.html')
def getPredictions(sp_length, sp_width, pt_length, pt_width):
    import pickle
    model = pickle.load(open('models/titanic_survival_ml_model.sav', 'rb'))
    scaler = pickle.load(open('models/scaler.sav', 'rb'))
    prediction = model.predict(scaler.transform([[sp_length, sp_width, pt_length, pt_width]]))
    if prediction is not None:
        return prediction
    else:
        return 'error'

def result(request):
    sp_length = float(request.GET['sp_length'])
    sp_width = float(request.GET['sp_width'])
    pt_length = float(request.GET['pt_length'])
    pt_width = float(request.GET['pt_width'])
    result = getPredictions(sp_length, sp_width, pt_length, pt_width)
    return render(request, 'result.html', {'result':result[0]})
def predict_view(request):
    if request.method == 'POST':
        # Get data from the POST request
        sp_length = float(request.POST.get('sp_length'))
        sp_width = float(request.POST.get('sp_width'))
        pt_length = float(request.POST.get('pt_length'))
        pt_width = float(request.POST.get('pt_width'))

        # Here, include your ML model prediction logic
        # For the sake of this example, let's assume the prediction is a string
        prediction = getPredictions(sp_length, sp_width, pt_length, pt_width)  # Replace with your actual prediction logic

        # Return the prediction as a JSON response
        return JsonResponse({'prediction': prediction})

    return render(request, 'index.html')