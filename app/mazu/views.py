from django.shortcuts import render
from django import forms
from django.http import HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.http.response import JsonResponse
from django.core import serializers
from django.db import Error

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

import json
import re
import os

from .models import Prompt, Last


#########
# Forms #
#########

class LoginForm(forms.Form):
    username = forms.CharField(
        label='',
        max_length=50,
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'type': 'text',
                'placeholder': 'Användarnamn'
            }
        )
    )
    password = forms.CharField(
        label='',
        max_length=50,
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'type': 'password',
                'placeholder': 'Lösenord'
            }
        )
    )


class MessageForm(forms.Form):
    message = forms.CharField(
        label='',
        max_length=100,
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'type': 'text',
                'placeholder': 'Säg något...',
            }
        )
    )


#########
# Views #
#########

def index(request):
    form = MessageForm()

    if "prompts" not in request.session:
        request.session["prompts"] = []

    return render(request, 'mazu/index.html', {
        "form": form,
        "prompts": request.session["prompts"],
    })


def message(request):
    if request.method == 'POST' and request.user.is_authenticated:
        form = MessageForm(request.POST)

        # Check if form data is valid
        if form.is_valid():
            # Process the data in form.cleaned_data
            prompt = form.cleaned_data['message']
            print(f"\nForm Content: {prompt}\n")

            # Store the prompt in the session
            request.session["prompts"] += [prompt]

            # Save form content to database
            # Add to prompt_text in Prompt model
            try:
                p = Prompt(prompt_text=prompt)
                p.save()
            except Error as e:
                return JsonResponse({
                    "error": e,
                }, status=500)

            # Redirect to new URL
            return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:login'))


def weather(request):
    if request.method == 'POST' and request.user.is_authenticated:
        if request.POST.get('zero'):
            print(f"\nReceived: {request.POST['zero']}\n")
        elif request.POST.get('one'):
            print(f"\nReceived: {request.POST['one']}\n")
        # Redirect to new URL
        return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:login'))


def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)

        # Check if form data is valid
        if form.is_valid():
            # Attempt to sign user in
            username = form.cleaned_data['username']
            password = form. cleaned_data['password']
            user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("mazu:index"))
        else:
            return render(request, "mazu/login.html", {
                "form": form,
                "message": "Fel användarnamn / lösenord"
            })

    form = LoginForm()
    return render(request, "mazu/login.html", {
        "form": form
    })


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("mazu:index"))


#######
# API #
#######

@csrf_exempt
def api_mazu(request):
    if request.method == "POST":
        # Check if authorization bearer token is valid
        authorization = request.headers.get("Authorization")
        bearer = authorization.split(' ')[1]
        if bearer != os.environ.get("BEARER"):
            raise Http404("Error: request not authorized")

        # Only return new prompts that have not been sent before
        # Check if db has a record of the last prompt sent
        try:
            last = Last.objects.all().last()
            if last is None:
                # db is empty, set 0 as reference value
                value = Last(last_object=0)
                value.save()
                # print(f"api_mazu initiated last_object: {l}")

        except Error as e:
            return JsonResponse({
                "error": e,
            }, status=500)

        # Acquire all the new values from db since last check
        try:
            # First check the id for the last acquired prompt
            last = Last.objects.order_by('-id')[:1].values()[0]["last_object"]
            # print(f"last:\n{last}")
        except Error as e:
            return JsonResponse({
                "error": e,
            }, status=500)

        # Acquire all prompts created after the previously last prompt
        try:
            # But first check if there are any new prompts
            if Prompt.objects.filter(id__gt=last).exists():
                data = Prompt.objects.filter(id__gt=last).order_by("id")
                print(f"Acquired new data: \n{data}")
            # if Prompt.objects.filter(id__gt=0).exists():
            #     data = Prompt.objects.filter(id__gt=0).order_by("id")
            #     print(f"Acquired new data:\n{data}")
            else:
                # Return an empty list
                print("No new data to acquire, api_mazu returning empty list of prompts")
                return JsonResponse({
                    "prompts": [],
                }, status=200)

        except Error as e:
            return JsonResponse({
                "error": e,
            }, status=500)

        # save the new last object's id to db, if there is one
        try:
            if len(data) > 0:
                new_last = data.values()[len(data) - 1]["id"]
                # print(data.values()[len(data) - 1])
                nl = Last(last_object=new_last)
                nl.save()
                # print("Saved new id for last prompt")
        except Error as e:
            return JsonResponse({
                "error": e,
            }, status=500)

        # Send json response to request
        serialized_data = serializers.serialize("json", data)
        data_list = json.loads(serialized_data)
        return JsonResponse({
            "prompts": data_list,
        }, status=200)

    raise Http404("Error: POST request required")
