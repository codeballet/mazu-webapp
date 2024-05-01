from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    pass

    def __str__(self):
        return (f"id: {self.id}, username: {self.username}")


class Message(models.Model):
    prompt = models.CharField(max_length=80)
    session_key = models.CharField(max_length=32)
    answer = models.CharField(max_length=1024)

    def __str__(self):
        return (
            f"id: {self.id}, \
            session_key: {self.session_key}, \
            prompt: {self.prompt}, \
            answer: {self.answer}"
        )


class Last(models.Model):
    last_object = models.BigIntegerField(default=0)

    def __str__(self):
        return (f"id: {self.id}, last_object: {self.last_object}")
