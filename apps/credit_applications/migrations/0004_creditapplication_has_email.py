# Generated by Django 4.2 on 2023-04-07 08:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_applications', '0003_alter_creditapplication_education_level_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='creditapplication',
            name='has_email',
            field=models.BooleanField(default=False),
        ),
    ]
