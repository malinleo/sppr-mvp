from django.db import models
from django.utils.translation import gettext_lazy as _

class CreditApplication(models.Model):
    """Credit application model."""
    applicant = models.ForeignKey(
        to="users.User",
        related_name="applications",
        on_delete=models.CASCADE,
    )
    limit_balance = models.PositiveIntegerField(
        verbose_name=_("Limit balance"),
    )
    gender = models.PositiveSmallIntegerField(
        verbose_name=_("Gender"),
        choices=((1, "Male"), (2, "Female")),
    )
    education_level = models.PositiveSmallIntegerField(
        verbose_name=_("Education level"),
        choices=(
            (1, 'Graduate school'),
            (2, 'University'),
            (3, 'High school'),
            (4, 'Others'),
        ),
    )
    marriage = models.PositiveSmallIntegerField(
        verbose_name=_("Marriage status"),
        choices=(
            (1, 'Married'),
            (2, 'Single'),
            (3, 'Others'),
        ),
    )
    age = models.PositiveSmallIntegerField(
        verbose_name=_("Age"),
    )
    pay_1 = models.PositiveIntegerField()
    pay_2 = models.PositiveIntegerField()
    pay_3 = models.PositiveIntegerField()
    pay_4 = models.PositiveIntegerField()
    pay_5 = models.PositiveIntegerField()
    pay_6 = models.PositiveIntegerField()

    bill_amt_1 = models.PositiveIntegerField()
    bill_amt_2 = models.PositiveIntegerField()
    bill_amt_3 = models.PositiveIntegerField()
    bill_amt_4 = models.PositiveIntegerField()
    bill_amt_5 = models.PositiveIntegerField()
    bill_amt_6 = models.PositiveIntegerField()
    
    pay_amt_1 = models.PositiveIntegerField()
    pay_amt_2 = models.PositiveIntegerField()
    pay_amt_3 = models.PositiveIntegerField()
    pay_amt_4 = models.PositiveIntegerField()
    pay_amt_5 = models.PositiveIntegerField()
    pay_amt_6 = models.PositiveIntegerField()

    approved = models.BooleanField(default=False)
