# Generated by Django 5.0.4 on 2024-04-17 15:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mazu', '0012_remove_last_last_sent_last_last_object'),
    ]

    operations = [
        migrations.AlterField(
            model_name='last',
            name='last_object',
            field=models.BigIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='prompt',
            name='created',
            field=models.BigIntegerField(default=0),
        ),
    ]
