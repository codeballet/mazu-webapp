from django.shortcuts import render
from django.http import HttpResponseRedirect
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

from .models import Message, Last, User, Vote
from .forms import LoginForm, MessageForm, RegisterForm


####################
# Global Variables #
####################

# Set whether or not mazu AIs are active
MAZU_ACTIVE = False


#########
# Views #
#########

def index(request):
    form = MessageForm()

    print("\nindex view starting:")
    print(f"Prompt: {request.session.get('prompt')}")
    print(f"Answer: {request.session.get('answer')}")
    print(f"Session key: {request.session.session_key}")

    # Initiate session variables
    request.session["mazu_active"] = MAZU_ACTIVE
    if "prompt" not in request.session:
        request.session["prompt"] = ''
    if "answer" not in request.session:
        request.session["answer"] = ''

    # Attempt to get the last existing entry from database
    try:
        if Message.objects.filter(session_key=request.session.session_key).exists():
            message = Message.objects.filter(
                session_key=request.session.session_key
            ).order_by('-id')[:1][0]

            print(f"\nindex view, latest message entry from db: \n{message}\n")

            # Set session variables
            request.session["prompt"] = message.prompt
            request.session["answer"] = message.answer
        else:
            print("index view found no entry for session_key in db")

            # Reset session variables
            request.session["prompt"] = ''
            request.session["answer"] = ''

    except IndexError as e:
        print(f"error: {e}")

        # Reset session variables
        request.session["prompt"] = ''
        request.session["answer"] = ''

    return render(request, 'mazu/index.html', {
        "form": form,
        "prompt": request.session.get("prompt"),
        "answer": request.session.get("answer"),
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
    # if request.method == 'POST' and request.user.is_authenticated:
    if request.method == 'POST':

        # Only move ahead if Mazu is active
        if not request.session.get("mazu_active"):
            messages.info(request, "Mazu är inte tillgänglig just nu.")
            messages.info(request, 'Kontakta Johan Stjernholm för nästa utställning med Mazu.')

            return HttpResponseRedirect(reverse('mazu:index'))

        form = MessageForm(request.POST)

        # Check if form data is valid
        if form.is_valid():
            # Process the data in form.cleaned_data
            prompt = form.cleaned_data['message']
            print(f"\nmessage view received Form Content: \n{prompt}\n")

            # Store the prompt in the session
            # request.session["prompts"] += [prompt]

            # Save form content and session_key to database
            try:
                new_message = Message(
                    prompt=prompt,
                    session_key=request.session.session_key,
                )
                new_message.save()

                # Set session with new prompt and empty answer
                request.session["prompt"] = prompt
                request.session["answer"] = ''

            except Error as e:
                print(f"message view error: \n{e}")
                return HttpResponseRedirect(reverse('mazu:index'))

            # Redirect to new URL
            return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:index'))


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
    # if request.method == 'POST' and request.user.is_authenticated:
    if request.method == 'POST':
        print(f"\nweather view received POST request: {request.POST}")

        # Only move ahead if Mazu is active
        if not request.session.get("mazu_active"):
            messages.info(request, "Mazu är inte tillgänglig just nu.")
            messages.info(request, 'Kontakta Johan Stjernholm för nästa utställning med Mazu.')

            return HttpResponseRedirect(reverse('mazu:index'))

        if Vote.objects.order_by("pk").last() is None:
            # No votes yet in db, initialise db with zeros
            print("No votes found in db")
            vote = Vote(zero=0, one=0)
            vote.save()

        if request.POST.get('zero'):
            print(f"\nweather view received: {request.POST['zero']}\n")
            # Update the latest vote 'zero' entry in the db with zero + 1
            vote = Vote.objects.order_by("pk").last()
            zero = vote.zero + 1
            pk = vote.pk
            print(f"zero: {zero}, pk: {pk}")
            Vote.objects.filter(pk=pk).update(zero=zero)
        if request.POST.get('one'):
            # Update the latest vote 'one' entry in the db with one + 1
            print(f"\nweather view received: {request.POST['one']}\n")
            vote = Vote.objects.order_by("pk").last()
            one = vote.one + 1
            pk = vote.pk
            print(f"one: {one}, pk: {pk}")
            Vote.objects.filter(pk=pk).update(one=one)

        # Redirect to index URL
        return HttpResponseRedirect(reverse('mazu:index'))

    return HttpResponseRedirect(reverse('mazu:index'))


######################
# External calls API #
######################

# Receiving API calls from index js script while waiting for answer.
def api_answer(request):
    if request.method == "POST":
        # Acquire latest message for a particular session
        try:
            message = Message.objects.filter(session_key=request.session.session_key).last()
            if message:
                request.session["answer"] = message.answer

                return JsonResponse({
                    "answer": message.answer,
                }, status=200)
            else:
                # return empty answer
                return JsonResponse({
                    "answer": '',
                }, status=200)
        except AttributeError:
            # Reset session values
            request.session["prompt"] = ''
            request.session["answer"] = ''

            # Return error code
            return JsonResponse({
                "error": "No answer found in db",
            }, status=500)

    print(f"\napi_answer received GET request: {request}\n")
    # TODO: find out why GET requests are sent to this view
    return JsonResponse({
        "answer": ""
    }, status=200)


# Respond with vote statistics to mazusea
@csrf_exempt
def api_sea(request):
    print("api_sea GET request received")
    # Check if request authorization bearer token is valid
    authorization = request.headers.get("Authorization")
    bearer = authorization.split(' ')[1]
    if bearer != os.environ.get("BEARER"):
        return JsonResponse({
            "error": "Request not authorized",
        }, status=401)

    # Aquire the Vote data
    vote = Vote.objects.order_by("pk").last()
    id = vote.id
    zeros = vote.zero
    ones = vote.one

    # Calculate the voting result
    if (zeros + ones) == 0:
        # no votes cast
        result = -1
    else:
        result = ones / (zeros + ones)

    # Reset the Vote db entry
    Vote.objects.filter(pk=id).update(zero=0, one=0)

    return JsonResponse({
        "vote": result,
    }, status=200)


# Respond with new user prompts to mazutalk
@csrf_exempt
def api_mazu(request):
    # Check if request authorization bearer token is valid
    authorization = request.headers.get("Authorization")
    bearer = authorization.split(' ')[1]
    if bearer != os.environ.get("BEARER"):
        return JsonResponse({
            "error": "Request not authorized",
        }, status=401)

    print("api_mazu authorization cleared!")

    # POST request
    if request.method == "POST":
        print("api_mazu POST request received")
        # Update database with answers from Mazu
        received_id = request.POST.get('id')
        received_answer = request.POST.get('answer')

        # Update message table in db with answer from Mazu
        try:
            Message.objects.filter(id=received_id).update(answer=received_answer)

            return JsonResponse({
                "message": "Successfully updated db",
            }, status=200)

        except IntegrityError as e:
            return JsonResponse({
                "error": e,
            }, status=500)

    # GET request, only return new prompts
    print("api_mazu GET request received")

    last = Last.objects.all().last()
    # Check if db has a record of the last prompt sent
    if last is None:
        # db is empty, set 0 as reference value in Last
        value = Last(last_prompt=0)
        value.save()

    # Acquire all the new entries from db since last check
    last = Last.objects.order_by('-id')[:1].values()[0]["last_prompt"]
    print(f"last prompt: {last}")

    # First check if there are any new prompts
    if Message.objects.filter(id__gt=last).exists():
        # Then get the entries
        data = Message.objects.filter(id__gt=last).order_by("id")
        print(f"api_mazu acquired new data: \n{data}")
    else:
        # Return an empty list if there are no new entries
        print("api_mazu has no new data to acquire, returning empty list of messages")
        data = []
        return JsonResponse({
            "messages": data,
        }, status=200)

    # Udate "last_prompt" in the Last db table, if there is one
    if len(data) > 0:
        new_last = data.values()[len(data) - 1]["id"]
        nl = Last(last_prompt=new_last)
        nl.save()

    # Send json response with the new entries
    serialized_data = serializers.serialize("json", data)
    data_list = json.loads(serialized_data)
    return JsonResponse({
        "messages": data_list,
    }, status=200)
