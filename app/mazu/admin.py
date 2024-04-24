from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Prompt, Last, User

# Register your models here.


class PromptAdmin(admin.ModelAdmin):
    list_display = ("id", "prompt_text")


class LastAdmin(admin.ModelAdmin):
    list_display = ("id", "last_object")


admin.site.register(Prompt, PromptAdmin)
admin.site.register(Last, LastAdmin)
admin.site.register(User, UserAdmin)
