# Generated by Django 4.1.13 on 2024-10-27 12:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Analysis', '0004_remove_analysis_shareholders_pattern_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysis',
            name='director_message',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
