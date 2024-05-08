from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Message, Last, User, Vote

# Register your models here.


class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session_key", "prompt", "answer")


class LastAdmin(admin.ModelAdmin):
    list_display = ("id", "last_prompt")

class VoteAdmin(admin.ModelAdmin):
    list_display = ("id", "zero", "one")


admin.site.register(Message, MessageAdmin)
admin.site.register(Last, LastAdmin)
admin.site.register(User, UserAdmin)
admin.site.register(Vote, VoteAdmin)
