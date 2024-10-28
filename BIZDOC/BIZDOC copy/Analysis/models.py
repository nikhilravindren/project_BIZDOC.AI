from django.db import models
import datetime
from django.contrib.auth.models import User
# Create your models here.

class company(models.Model):
    name = models.CharField(max_length=100, null=False, blank=False)
    sector = models.CharField(max_length=100, null=False, blank=False,default=1)


class sentiment(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    company = models.ForeignKey(company,on_delete=models.CASCADE)
    date = models.DateField(blank=False,default=datetime.datetime.now)
    news = models.TextField(max_length=1000, null=False, blank=False,default=1)
    sentiment = models.CharField(max_length=100, null=False, blank=False,default=1)
    confidence = models.FloatField(null=False, blank=False)


class Analysis(models.Model):
    date = models.DateField(blank=False,default=datetime.datetime.now)
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    company = models.ForeignKey(company,on_delete=models.CASCADE)
    summary = models.JSONField(null=True,blank=True)
    balance_sheet = models.JSONField(null=True,blank=True)
    director_message = models.JSONField(null=True,blank=True)


class shareholders_pattern(models.Model):
    company = models.ForeignKey(company,on_delete=models.CASCADE)
    holder = models.JSONField(null=True,blank=True)
    numbers = models.JSONField(null=True,blank=True)


class comparison(models.Model):
    company1 = models.ForeignKey(company,on_delete=models.CASCADE,related_name='comparison_as_company1')
    company2 = models.ForeignKey(company,on_delete=models.CASCADE,related_name='comparison_as_company2')
    report  =  models.JSONField(null=True,blank=True)
    date = models.DateField(blank=False,default=datetime.datetime.now)

class contanct(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    name = models.CharField(null=False,blank=False,max_length=225)
    mail = models.CharField(null=False,blank=False,max_length=225)
    message  = models.TextField(max_length=1000, null=False, blank=False)
    date = models.DateField(blank=False,default=datetime.datetime.now)



