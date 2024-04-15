from django.shortcuts import render
from django import forms
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http.response import JsonResponse
from django.core import serializers
import datetime

import json
import re

from .models import Prompt


# Forms
class MessageForm(forms.Form):
    message = forms.CharField(
        label='', 
        max_length=200, 
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'placeholder': 'Säg något...',
            }
        )
    )


# Views
def index(request):
    form = MessageForm()

    return render(request, 'mazu/index.html', {
        "form": form,
    })

def message(request):
    if request.method == 'POST':
        form = MessageForm(request.POST)

        # Check if form data is valid
        if form.is_valid():
            # Process the data in form.cleaned_data
            print(f"\nForm Content: {form.cleaned_data['message']}\n")

            # Get a unique timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

            # Save form content to database
            # Add to prompt_text in Prompt model
            p = Prompt(created = timestamp, prompt_text = form.cleaned_data['message'])
            p.save()
            
            # Redirect to new URL
            return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:index'))

def weather(request):
    if request.method == 'POST':
        if request.POST.get('zero'):
            print(f"\nReceived: {request.POST['zero']}\n")
        elif request.POST.get('one'):
            print(f"\nReceived: {request.POST['one']}\n")
        # Redirect to new URL
        return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:index'))

# API
def api_mazu(request):
    try:
        data = serializers.serialize("json", Prompt.objects.all().order_by("created"))
        list = json.loads(data)
        return JsonResponse({
            "prompts": list,
        }, status=200)
    except:
        print("\nError: could not acquire and send data from web app database\n")