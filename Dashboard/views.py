from django.shortcuts import render


# Create your views here.
def homepage(request):
    context = dict()
    return render(request, 'Dashboard/Homepage.html', context)
