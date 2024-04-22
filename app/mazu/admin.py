from django.contrib import admin

from .models import Prompt, Last

# Register your models here.


class PromptAdmin(admin.ModelAdmin):
    list_display = ("id", "prompt_text")


class LastAdmin(admin.ModelAdmin):
    list_display = ("id", "last_object")


admin.site.register(Prompt, PromptAdmin)
admin.site.register(Last, LastAdmin)
