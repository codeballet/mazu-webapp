from django import forms


class LoginForm(forms.Form):
    username = forms.CharField(
        label='',
        max_length=50,
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'type': 'text',
                'placeholder': 'Användarnamn...',
                'autofocus': 'autofocus'
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
                'placeholder': 'Lösenord...'
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
                'placeholder': 'Skriv till Mazu...',
            }
        )
    )


class RegisterForm(forms.Form):
    username = forms.CharField(
        label='',
        max_length=50,
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'type': 'text',
                'placeholder': 'Användarnamn...',
                'autofocus': 'autofocus'
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
                'placeholder': 'Lösenord...'
            }
        )
    )
    confirmation = forms.CharField(
        label='',
        max_length=50,
        widget=forms.TextInput(
            attrs={
                'class': 'textfield',
                'type': 'password',
                'placeholder': 'Skriv lösenordet igen...'
            }
        )
    )
