# Generated by Django 4.1.13 on 2024-10-08 06:36

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('Analysis', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Analysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('summary', models.JSONField(blank=True, null=True)),
                ('shareholders_pattern', models.JSONField(blank=True, null=True)),
                ('balance_sheet', models.JSONField()),
                ('company', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='Analysis.company')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
