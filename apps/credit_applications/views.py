from pathlib import Path

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views.generic import (
    CreateView,
    DetailView,
    FormView,
    ListView,
)

from rest_framework import response, serializers, status
from rest_framework.generics import RetrieveAPIView
from rest_framework.permissions import IsAdminUser

from apps.core.mixins import UserContextMixin

from .forms import (
    CreditApplicationCreateForm,
    CreditApplicationForm,
    CreditApplicationListForm,
)
from .models import CreditApplication
from .predictions import (
    credit_application_to_dataframe,
    load_classifier,
    load_label_encoders,
)


class ApplicationCreateView(LoginRequiredMixin, UserContextMixin, CreateView):
    """Create application."""
    form_class = CreditApplicationCreateForm
    template_name = "credit_applications/create.html"

    def get_success_url(self) -> str:
        return reverse_lazy("applications-detail", args=(self.object.pk,))

    def form_valid(self, form: CreditApplicationCreateForm):
        self.object = form.save(commit=False)
        self.object.applicant = self.request.user
        self.object.save()
        return super().form_valid(form)


class ApplicationDetailView(LoginRequiredMixin, UserContextMixin, DetailView):
    """Detail info."""
    form_class = CreditApplicationForm
    template_name = "credit_applications/detail.html"
    queryset = CreditApplication.objects.all()

    def get_queryset(self):
        if self.request.user.is_staff:
            return super().get_queryset()
        return super().get_queryset().filter(applicant=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["form"] = self.form_class(instance=kwargs.get("object"))
        return context


class ApplicationListView(LoginRequiredMixin, UserContextMixin, ListView):
    """List all applications."""
    form_class = CreditApplicationListForm
    queryset = CreditApplication.objects.all()

    def get_queryset(self):
        if self.request.user.is_staff:
            return super().get_queryset()
        return super().get_queryset().filter(
            applicant=self.request.user,
        ).order_by("approved")


class ApplicationApproveView(LoginRequiredMixin, UserContextMixin, FormView):
    """Approve application."""
    queryset = CreditApplication.objects.all()

    def get_success_url(self) -> str:
        return reverse_lazy("applications-detail", args=(self.object.pk,))

    def get_queryset(self):
        if self.request.user.is_staff:
            return self.queryset
        return self.queryset.filter(applicant=self.request.user)
    
    def post(self, request, pk, *args, **kwargs):
        self.object = get_object_or_404(self.get_queryset(), pk=pk)
        self.object.approved = True
        self.object.save()
        return HttpResponseRedirect(self.get_success_url())


class CreditPredictionSerializer(serializers.Serializer):
    """Serializer for CreditApplication."""
    prediction = serializers.BooleanField()


class CreditScoringAPIView(RetrieveAPIView):
    """API view for credit scoring prediction."""
    classifier = load_classifier()
    # label_encoders = load_label_encoders(Path("dataset_2.csv"))
    permission_classes = (IsAdminUser,)
    queryset = CreditApplication.objects.all()
    serializer_class = CreditPredictionSerializer 

    def get(self, request, pk, *args, **kwargs):
        """Predict if given application is good or bad."""
        application: CreditApplication = self.get_object()
        data = credit_application_to_dataframe(application)
        prediction = self.classifier.predict([data.iloc[0].to_list()])
        serializer = self.get_serializer(data={"prediction": prediction[0]})
        serializer.is_valid(raise_exception=True)
        return response.Response(
            data=serializer.validated_data,
            status=status.HTTP_200_OK,
        )

