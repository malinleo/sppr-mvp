from django.urls import reverse_lazy
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from .predictions import load_classifier, credit_application_to_dataframe, load_label_encoders
from django.views.generic import CreateView, TemplateView, DeleteView, DetailView, UpdateView, ListView, FormView, RedirectView
from rest_framework.views import APIView
from rest_framework.generics import RetrieveAPIView
from rest_framework import response, status
from pathlib import Path
from rest_framework import serializers
from .forms import CreditApplicationCreateForm, CreditApplicationForm, CreditApplicationListForm
from .models import CreditApplication
from django.contrib.auth.mixins import LoginRequiredMixin
from apps.core.mixins import UserContextMixin


class ApplicationCreateView(LoginRequiredMixin, UserContextMixin, CreateView):
    """Create application."""
    form_class = CreditApplicationCreateForm
    template_name = "credit_applications/create.html"

    def get_success_url(self) -> str:
        return reverse_lazy("applications-detail", args=(self.object.pk,))

    def form_valid(self, form):
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
        return super().get_queryset().filter(applicant=self.request.user)


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
    label_encoders = load_label_encoders(Path("dataset_without_encoding.csv"))
    queryset = CreditApplication.objects.all()
    serializer_class = CreditPredictionSerializer

    def get(self, request, pk, *args, **kwargs):
        """Predict if given application is good or bad."""
        application: CreditApplication = self.get_object()
        data = credit_application_to_dataframe(application)
        transformed_data = []
        for col, label_encoder in zip(data.columns, self.label_encoders):
            if data[col].dtype == "object":
                transformed_data.extend(label_encoder.transform(data[col]))
            else:
                transformed_data.extend(data[col])
        prediction = self.classifier.predict([transformed_data])
        serializer = self.get_serializer(data={"prediction": prediction[0]})
        serializer.is_valid(raise_exception=True)
        return response.Response(
            data=serializer.validated_data,
            status=status.HTTP_200_OK,
        )

