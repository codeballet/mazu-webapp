from django.db import models

# Create your models here.

class Prompt(models.Model):
    prompt_text = models.CharField(max_length=80)
    created = models.CharField(max_length=80)

    def __str__(self):
        return (f"{self.created}: {self.prompt_text}")