# Generated by Django 4.1.13 on 2024-10-08 11:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Analysis', '0002_analysis'),
    ]

    operations = [
        migrations.AlterField(
            model_name='analysis',
            name='balance_sheet',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
