from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Message, Last, User

# Register your models here.


class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "session_key", "prompt_text", "answer")


class LastAdmin(admin.ModelAdmin):
    list_display = ("id", "last_object")


admin.site.register(Message, MessageAdmin)
admin.site.register(Last, LastAdmin)
admin.site.register(User, UserAdmin)
