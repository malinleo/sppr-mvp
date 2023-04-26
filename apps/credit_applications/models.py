from django.db import models


class CreditApplication(models.Model):
    """Credit application model."""
    applicant = models.ForeignKey(
        to="users.User",
        related_name="applications",
        on_delete=models.CASCADE,
    )
    gender = models.CharField(
        max_length=1,
        choices=(("F", "Female"), ("M", "Male")),
    )
    own_car = models.BooleanField()
    own_realty = models.BooleanField()
    children_count = models.PositiveSmallIntegerField()
    annual_income = models.PositiveIntegerField()
    income_type = models.CharField(
        choices=(
            ('Commercial associate', 'Commercial associate'),
            ('Pensioner', 'Pensioner'),
            ('State servant', 'State servant'),
            ('Student', 'Student'),
            ('Working', 'Working'),
        ),
    )
    education_level = models.CharField(
        choices=(
            ('Academic degree', 'Academic degree'),
            ('Higher education', 'Higher education'),
            ('Incomplete higher', 'Incomplete higher'),
            ('Lower secondary', 'Lower secondary'),
            ('Secondary / secondary special', 'Secondary / secondary special'),
        ),
    )
    family_status = models.CharField(
        choices=(
            ('Civil marriage', 'Civil marriage'),
            ('Married', 'Married'),
            ('Separated', 'Separated'),
            ('Single / not married', 'Single / not married'),
            ('Widow', 'Widow'),
        ),
    )
    housing_type = models.CharField(
        choices=(
            ('Co-op apartment', 'Co-op apartment'),
            ('House / apartment', 'House / apartment'),
            ('Municipal apartment', 'Municipal apartment'),
            ('Office apartment', 'Office apartment'),
            ('Rented apartment', 'Rented apartment'),
            ('With parents', 'With parents'),
        ),
    )
    bday_shift = models.IntegerField()
    days_employed = models.IntegerField()
    has_mobile = models.BooleanField()
    has_work_phone = models.BooleanField()
    has_phone = models.BooleanField()
    has_email = models.BooleanField(default=False)
    occupation_type = models.CharField(null=True)
    family_members_count = models.PositiveSmallIntegerField()
    approved = models.BooleanField(default=False)
