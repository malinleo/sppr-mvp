from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.views import LoginView as DjangoLoginView
from django.shortcuts import resolve_url
from django.urls import reverse_lazy
from django.views.generic import (
    CreateView,
    DeleteView,
    DetailView,
    TemplateView,
    UpdateView,
)

from apps.core.mixins import UserContextMixin

from .forms import AuthForm


class RegisterView(UserContextMixin, CreateView):
    """Register user."""
    success_url = reverse_lazy('login')
    template_name = "register.html"
    form_class = AuthForm


class LoginView(UserContextMixin, DjangoLoginView):
    """Login user."""
    template_name = "login.html"
    form_class = AuthenticationForm
    next_page = reverse_lazy("applications-list")
