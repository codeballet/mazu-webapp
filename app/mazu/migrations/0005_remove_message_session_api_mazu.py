# Generated by Django 5.0.4 on 2024-05-01 10:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mazu', '0004_message_session_api_mazu'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='message',
            name='session_api_mazu',
        ),
    ]
