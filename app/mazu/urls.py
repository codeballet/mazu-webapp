from django.urls import path
from . import views

app_name = 'mazu'
urlpatterns = [
    path('', views.index, name='index'),
    path('message/', views.message, name='message'),
    path('weather/', views.weather, name='weather'),
    path('api_mazu/', views.api_mazu, name='api_mazu'),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]
