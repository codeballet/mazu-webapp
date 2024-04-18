from django.shortcuts import render
from django import forms
from django.http import HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.http.response import JsonResponse
from django.core import serializers
import datetime

import json
import re

from .models import Prompt, Last


#########
# Forms #
#########
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

#########
# Views #
#########
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

            # # Get a unique timestamp
            # # timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            # timestamp = int(datetime.datetime.now().strftime('%d%H%M%S%f'))
            # print(f"Timestamp: {timestamp}, {type(timestamp)}")

            # Save form content to database
            # Add to prompt_text in Prompt model
            try:
                # p = Prompt(created = timestamp, prompt_text = form.cleaned_data['message'])
                p = Prompt(prompt_text = form.cleaned_data['message'])
                p.save()
            except:
                raise Http404("Cannot save to database")
            
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


#######
# API #
#######
def api_mazu(request):
    # Only return new prompts that have not been sent before
    try:
        # Check if db is empty
        last = Last.objects.all().last()
        if last == None:
            # # db is empty, set current time as a reference value in db
            # timestamp = int(datetime.datetime.now().strftime('%d%H%M%S%f'))
            # l = Last(last_object=timestamp)
            # l.save()
            # print(f"api_mazu initiated last_object: {l}")

            # db is empty, set 0 as reference value
            l = Last(last_object=0)
            l.save()
            print(f"api_mazu initiated last_object: {l}")

    except:
        print("api_mazu failed to initiate last_object")

    # Acquire all the new values from db since last check
    try:
        # First check the id for the last acquired prompt
        last = Last.objects.order_by('-id')[:1].values()[0]["last_object"]
        print(f"last:\n{last}")
    except:
        print("api_mazu failed to aquire the last object")

    try:
        # # Then acquire all prompts created after the previously last prompt
        # # But first check if there are any new prompts
        # if Prompt.objects.filter(created__gt=last).exists():
        #     data = Prompt.objects.filter(created__gt=last).order_by("created")
        #     print(f"Acquired new data:\n{data}")

        if Prompt.objects.filter(id__gt=last).exists():
            data = Prompt.objects.filter(id__gt=last).order_by("id")
            print(f"Acquired new data:\n{data}")
        else:
            # Return an empty list
            print("No new data to acquire, api_mazu returning empty list of prompts")
            return JsonResponse({
                "prompts": [],
            }, status=200)

    except:
        raise Http404("api_mazu could not acquire prompts from db")

    try:
        # # save the new last object's time stamp to db, if there is one
        # if len(data) > 0:
        #     new_last = data.values()[len(data) - 1]["created"]
        #     print(data.values()[len(data) - 1])
        #     nl = Last(last_object=new_last)
        #     nl.save()
        #     print("Saved new timestamp for last prompt")

        # save the new last object's id to db, if there is one
        if len(data) > 0:
            new_last = data.values()[len(data) - 1]["id"]
            print(data.values()[len(data) - 1])
            nl = Last(last_object=new_last)
            nl.save()
            print("Saved new timestamp for last prompt")
    except:
        raise Http404("api_mazu could not save last_object to db")

    # Send json response to request
    serialized_data = serializers.serialize("json", data)
    data_list = json.loads(serialized_data)
    return JsonResponse({
        "prompts": data_list,
    }, status=200)
