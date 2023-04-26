from django import forms
from . import models
from django.contrib.auth.forms import UserCreationForm


class AuthForm(UserCreationForm):

    class Meta:
        model = models.User
        fields = (
            "first_name",
            "last_name",
            "email",
        )
