from django.views.generic.base import ContextMixin


class UserContextMixin(ContextMixin):
    """Adds user to template context."""

    def get_context_data(self, *args, **kwargs):
        """Add user to context."""
        context = super().get_context_data(*args, **kwargs)
        context["user"] = self.request.user
        return context
