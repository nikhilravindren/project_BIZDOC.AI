"""
URL configuration for BIZDOC project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Analysis.views import home,form_submit,user_user_login,create_ac,chat,user_logout,dashboard,chating,shareholding,director_message
from cadmin.views import DBadmin
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',home,name='home'),
    path('form',form_submit,name='form'),
    path('login',user_user_login,name='login'),
    path('register',create_ac,name='register'),
    path('chat',chat,name='chat'),
    path('logout',user_logout,name='logout'),
    path('dashboard',dashboard,name='dashboard'),
    path('chating',chating,name='chating'),
    path('shareholding',shareholding,name='shareholding'),
    path('director_message',director_message,name='director_message'),

    path('DBadmin',DBadmin,name='DBadmin'),

]
