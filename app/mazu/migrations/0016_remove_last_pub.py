# Generated by Django 5.0.4 on 2024-04-17 20:07

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mazu', '0015_rename_published_last_pub'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='last',
            name='pub',
        ),
    ]
