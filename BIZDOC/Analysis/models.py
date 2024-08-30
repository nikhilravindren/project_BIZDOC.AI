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
