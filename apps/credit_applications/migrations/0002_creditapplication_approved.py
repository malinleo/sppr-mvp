# Generated by Django 4.2 on 2023-04-06 10:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_applications', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='creditapplication',
            name='approved',
            field=models.BooleanField(default=False),
        ),
    ]
