from django.db import models

# Create your models here.

class Prompt(models.Model):
    prompt_text = models.CharField(max_length=80)
    created = models.BigIntegerField(default=0)

    def __str__(self):
        return (f"{self.created}, {self.prompt_text}")

class Last(models.Model):
    last_object = models.BigIntegerField(default=0)

    def __str__(self):
        return (f"{self.last_object}")