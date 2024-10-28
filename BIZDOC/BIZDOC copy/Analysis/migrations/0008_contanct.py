# Generated by Django 4.1.13 on 2024-10-28 07:21

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Analysis', '0007_comparison'),
    ]

    operations = [
        migrations.CreateModel(
            name='contanct',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=225)),
                ('mail', models.CharField(max_length=225)),
                ('message', models.TextField(max_length=1000)),
                ('date', models.DateField(default=datetime.datetime.now)),
            ],
        ),
    ]