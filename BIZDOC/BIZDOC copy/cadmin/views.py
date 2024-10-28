from django.shortcuts import render
from Analysis.models import company,sentiment
import requests

# Create your views here.

def DBadmin(request):
    return render(request, 'cadmin.html')