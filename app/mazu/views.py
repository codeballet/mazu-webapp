from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.http.response import JsonResponse
from django.core import serializers
from django.db import Error, IntegrityError
from django.contrib import messages

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

import json
import re
import os

from .models import Message, Last, User
from .forms import LoginForm, MessageForm, RegisterForm


#########
# Views #
#########

def index(request):
    form = MessageForm()
    print(f"session: {request.session.session_key}")

    # Initiate "prompt" and "answer" in session
    if "prompt" not in request.session:
        request.session["prompt"] = ''
    if "answer" not in request.session:
        request.session["answer"] = ''

    # Get the last existing entry from database
    try:
        message = Message.objects.filter(
            session_key=request.session.session_key
            ).order_by('-id')[:1][0]
        print(f"\nfrom db:\n{message}\n")
    except:
        print("No session_key recorded in db yet")

    # Send answer to template
    try:
        if message.answer:
            request.session["prompt"] = message.prompt_text
            request.session["answer"] = message.answer
    except:
        print("No answer from last prompt yet")

    return render(request, 'mazu/index.html', {
        "form": form,
    })


def about(request):
    return render(request, 'mazu/about.html')


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
            messages.success(request, 'Välkommen %s!' % user.username)
            messages.success(request, 'Sänd Mazu meddelenaden och lyssna noga i lokalen där du är.')
            messages.success(request, 'Mazu kommer att svara dig.')
            return HttpResponseRedirect(reverse("mazu:index"))
        else:
            messages.info(request, 'Fel användarnamn / lösenord')
            return render(request, "mazu/login.html", {
                "form": form,
            })

    form = LoginForm()
    return render(request, "mazu/login.html", {
        "form": form
    })


def logout_view(request):
    logout(request)
    messages.success(request, 'Farväl, min vän!')
    return HttpResponseRedirect(reverse("mazu:index"))


def message(request):
    # TODO: restore authentication
    # if request.method == 'POST' and request.user.is_authenticated:
    if request.method == 'POST':
        form = MessageForm(request.POST)

        # Check if form data is valid
        if form.is_valid():
            # Process the data in form.cleaned_data
            prompt = form.cleaned_data['message']
            print(f"\nForm Content: {prompt}\n")

            # Store the prompt in the session
            # request.session["prompts"] += [prompt]

            # Save form content and session_key to database
            try:
                new_message = Message(
                    prompt_text=prompt,
                    session_key=request.session.session_key,
                )
                new_message.save()

                # Set session with new prompt and empty answer
                request.session["prompt"] = prompt
                request.session["answer"] = ''

            except Error as e:
                return JsonResponse({
                    "error": e,
                }, status=500)

            # Redirect to new URL
            return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:login'))


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)

        # Check if form data is valid
        if form.is_valid():
            # Attempt to sign user in
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            confirmation = form.cleaned_data['confirmation']

            # Ensure password matches confirmation
            if password != confirmation:
                messages.info(request, "Lösenorden stämmer inte!")
                return render(request, 'mazu/register.html', {
                    'form': form,
                })

            # Attempt to create new user
            try:
                user = User.objects.create_user(
                    username=username,
                    email=None,
                    password=password
                )
                user.save()
            except IntegrityError:
                messages.info(request, "Det användarnamet är redan taget")
                return render(request, 'mazu/register.html', {
                    'form': form,
                })

            # Login user
            login(request, user)

            # Load homepage
            messages.success(request, 'Välkommen %s!' % user.username)
            return HttpResponseRedirect(reverse('mazu:index'))

    # Instantiate the RegisterForm and load register page
    form = RegisterForm()
    return render(request, 'mazu/register.html', {
        'form': form,
    })


def weather(request):
    if request.method == 'POST' and request.user.is_authenticated:
        if request.POST.get('zero'):
            print(f"\nReceived: {request.POST['zero']}\n")
        elif request.POST.get('one'):
            print(f"\nReceived: {request.POST['one']}\n")
        # Redirect to new URL
        return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:login'))


#######
# API #
#######

@csrf_exempt
def api_mazu(request):
    # Check if request authorization bearer token is valid
    authorization = request.headers.get("Authorization")
    bearer = authorization.split(' ')[1]
    if bearer != os.environ.get("BEARER"):
        raise Http404("Error: request not authorized")

    # Deal with POST requests
    if request.method == "POST":
        # Update database with answers from Mazu
        received_id = request.POST.get('id')
        received_prompt = request.POST.get('prompt')
        received_answer = request.POST.get('answer')

        # Update db with answer from Mazu
        try:
            Message.objects.filter(id=received_id).update(answer=received_answer)
            
            # Update session variables
            request.session["prompt"] = received_prompt
            request.session["answer"] = received_answer

            form = MessageForm
            return render(request, 'mazu/index.html', {
                "form": form,
            })

            return JsonResponse({
                "message": "successfully updated db",
            }, status=201)

        except IntegrityError as e:
            return JsonResponse({
                "error": e,
            }, status=500)


    # Deal with GET requests:
    # Only return new prompts that have not been sent before
    # Check if db has a record of the last prompt sent
    print("GET request received")
    try:
        last = Last.objects.all().last()
        if last is None:
            # db is empty, set 0 as reference value in Last
            value = Last(last_object=0)
            value.save()
            # print(f"api_mazu initiated last_object: {l}")

    except Error as e:
        return JsonResponse({
            "error": e,
        }, status=500)

    # Acquire all the new entries from db since last check
    try:
        # First check the entry id for the last acquired prompt
        last = Last.objects.order_by('-id')[:1].values()[0]["last_object"]
        # print(f"last:\n{last}")
    except Error as e:
        return JsonResponse({
            "error": e,
        }, status=500)

    # Acquire all entries created after the previously last prompt
    try:
        # First check if there are any new prompts
        if Message.objects.filter(id__gt=last).exists():
            # Then get the entries
            data = Message.objects.filter(id__gt=last).order_by("id")
            print(f"Acquired new data: \n{data}")
        else:
            # Return an empty list if there are no new entries
            print("No new data to acquire, api_mazu returning empty list of messages")
            return JsonResponse({
                "messages": [],
            }, status=200)

    except Error as e:
        return JsonResponse({
            "error": e,
        }, status=500)

    # save the new last object's id to db, if there is one
    try:
        if len(data) > 0:
            new_last = data.values()[len(data) - 1]["id"]
            nl = Last(last_object=new_last)
            nl.save()
    except Error as e:
        return JsonResponse({
            "error": e,
        }, status=500)

    # Send json response with the new entries
    serialized_data = serializers.serialize("json", data)
    data_list = json.loads(serialized_data)
    return JsonResponse({
        "messages": data_list,
    }, status=200)


# Return session variable "answer"
def api_answer(request):
    message = Message.objects.filter(session_key=request.session.session_key).last()
    return JsonResponse({
        "answer": message.answer,
    }, status=200)