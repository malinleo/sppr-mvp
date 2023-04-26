from collections import namedtuple

from django.views.generic import TemplateView
from .mixins import UserContextMixin

Changelog = namedtuple("Changelog", ["name", "text", "version", "open_api_ui"])


class IndexView(UserContextMixin, TemplateView):
    """Class-based view for main page."""

    template_name = "index.html"
