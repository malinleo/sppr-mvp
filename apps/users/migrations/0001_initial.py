# Generated by Django 3.2.4 on 2021-06-18 07:39

import apps.users.models
import config.settings.common.paths
import django.contrib.postgres.fields.citext
import django.core.validators
import django_extensions.db.fields
import imagekit.models.fields
from django.contrib.postgres.operations import CITextExtension
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        CITextExtension(),
        migrations.CreateModel(
            name="User",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("password", models.CharField(max_length=128, verbose_name="password")),
                ("last_login", models.DateTimeField(blank=True, null=True, verbose_name="last login")),
                ("is_superuser", models.BooleanField(default=False, help_text="Designates that this user has all permissions without explicitly assigning them.", verbose_name="superuser status")),
                ("created", django_extensions.db.fields.CreationDateTimeField(auto_now_add=True, verbose_name="created")),
                ("modified", django_extensions.db.fields.ModificationDateTimeField(auto_now=True, verbose_name="modified")),
                ("first_name", models.CharField(blank=True, max_length=30, verbose_name="First name")),
                ("last_name", models.CharField(blank=True, max_length=30, verbose_name="Last name")),
                ("email", django.contrib.postgres.fields.citext.CIEmailField(max_length=254, unique=True, verbose_name="Email address")),
                ("is_staff", models.BooleanField(default=False, help_text="Designates whether the user can log into this admin site.", verbose_name="Staff status")),
                ("is_active", models.BooleanField(default=True, help_text="Designates whether this user should be treated as active.", verbose_name="Active")),
                ("avatar", imagekit.models.fields.ProcessedImageField(blank=True, max_length=512, null=True, upload_to=config.settings.common.paths._default_media_path, verbose_name="Avatar")),
                ("groups", models.ManyToManyField(blank=True, help_text="The groups this user belongs to. A user will get all permissions granted to each of their groups.", related_name="user_set", related_query_name="user", to="auth.Group", verbose_name="groups")),
                ("user_permissions", models.ManyToManyField(blank=True, help_text="Specific permissions for this user.", related_name="user_set", related_query_name="user", to="auth.Permission", verbose_name="user permissions")),
            ],
            options={
                "verbose_name": "User",
                "verbose_name_plural": "Users",
            },
            managers=[
                ("objects", apps.users.models.UserManager()),
            ],
        ),
    ]
