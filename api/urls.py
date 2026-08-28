from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.index, name='index'),
    re_path(r'^predict/$', views.predict_digit, name='predict_digit'),
    re_path(r'^feedback/$', views.submit_feedback, name='submit_feedback'),
]
