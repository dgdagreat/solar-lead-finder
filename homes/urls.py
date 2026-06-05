from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('search/', views.search_homes, name='search_homes'),
    path('home/<int:home_id>/', views.home_detail, name='home_detail'),
    path('home/<int:home_id>/reprocess/', views.reprocess_home, name='reprocess_home'),
    path('home/<int:home_id>/confirm-solar/', views.confirm_solar, name='confirm_solar'),
    path('home/<int:home_id>/confirm-no-solar/', views.confirm_no_solar, name='confirm_no_solar'),
    path('process-all/', views.process_all, name='process_all'),
]
