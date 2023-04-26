from django.urls import path
from . import views

urlpatterns = [
    path("create/", views.ApplicationCreateView.as_view(), name="applications-create"),
    path("<int:pk>/", views.ApplicationDetailView.as_view(), name="applications-detail"),
    path("", views.ApplicationListView.as_view(), name="applications-list"),
    path("approve/<int:pk>/", views.ApplicationApproveView.as_view(), name="applications-approve"),
    path("<int:pk>/predict/", views.CreditScoringAPIView.as_view(), name="predict"),
]
