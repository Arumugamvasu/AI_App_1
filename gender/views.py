from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from mfcc_feature import mfcc
from model_gen import get_model
# Create your views here.

def home2(request):
    #return HttpResponse('Hello Arumugam')
   # return render(request,'index.html')
   return render(request,'index.html',{'name':'Arumugam_kumar'})
class Home(TemplateView):
    template_name = 'home.html'

def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        #context['url'] = fs.url(name)
        file_dir='./media/'+name
        model = get_model.create_model()
    # load the saved/trained weights
        #model.load_weights(model_name)
        model.load_weights('./model/model.h5')
  


        features = mfcc.extract_mfcc(file_dir, mel=True).reshape(1, -1)
        # predict the gender!
        male_prob = model.predict(features)[0][0]
        print(male_prob)
        female_prob = 1 - male_prob
        gender = "Male" if male_prob > female_prob else "Female"
        # show the result!
        #   print("Result:", gender)
        if gender == 'Female':
           result_prob=female_prob*100
           a='Female'
        else:
            result_prob=male_prob*100
            a='Male'



        context['data'] = gender
    return render(request, 'upload.html', context)