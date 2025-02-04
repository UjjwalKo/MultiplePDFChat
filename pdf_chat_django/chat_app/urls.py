from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process_pdfs/', views.process_pdfs, name='process_pdfs'),
    path('chat/', views.chat, name='chat'),
]