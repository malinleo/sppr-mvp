from django import forms
from . import models


class CreditApplicationCreateForm(forms.ModelForm):
    """Credit application create form."""

    class Meta:
        model = models.CreditApplication
        exclude = ("applicant", "approved")


class CreditApplicationForm(forms.ModelForm):
    """Credit application form."""

    class Meta:
        model = models.CreditApplication
        fields = "__all__"


class CreditApplicationListForm(forms.ModelForm):
    """Credit application list form."""

    class Meta:
        model = models.CreditApplication
        fields = (
            "id",
            "applicant",
            "approved",
        )
