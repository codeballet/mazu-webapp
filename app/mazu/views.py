from django.shortcuts import render
from django import forms
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http.response import JsonResponse
import json
import re



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

            # Save form content to database
            
            
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
    return JsonResponse({
        "message": "Mazu says hello!",
    }, status=200)