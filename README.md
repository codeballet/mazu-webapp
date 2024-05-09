# Mazu Project Web App
The Mazu Project Web App is the public web app interface for the "Mazu: Water Legends" Performance Art project by Johan Stjernholm.

The Web App serves as an interface whereby audience members can interact with Artificial Intelligences and participate in generating live content for the show.

The project is sponsored by Rum för Dans and the Kungsbacka Theatre in Sweden. Premiere performance is 28 June 2024.

## Setting up the project
### Prerequisites
- A registered domain name.
- A DNS record with `your_domain.com` pointing to your proxy server’s public IP address.
- An SSH connection to the server.

### Environment variables
At the root of the project, on the server, create a `.env.prod` file specifying the following environment variables:
```
DEBUG=0
SECRET_KEY=a_secret_key
DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1] example.com
CSRF_TRUSTED_ORIGINS=https://example.com
BEARER=a_bearer_token
DJANGO_SUPERUSER_PASSWORD=password
DJANGO_SUPERUSER_EMAIL=example@mail.com
DJANGO_SUPERUSER_USERNAME=username
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
MAZU_ACTIVE=<False or True>
```
Those variables are secret. Do not include them in a public git repository or similar.

### Convenient commands for the production server
The below commands are for setting up the Django app on the production server.

Run the production version:
```
sudo docker compose -f docker-compose.prod.yml up --build -d
```
Make migrations:
```
sudo docker compose -f docker-compose.prod.yml exec web python manage.py makemigrations 

```
Migrate database:
```
sudo docker compose -f docker-compose.prod.yml exec web python manage.py migrate --noinput
```
Collect static files:
```
sudo docker compose -f docker-compose.prod.yml exec web python manage.py collectstatic
```

### Create `superuser` programmatically
Create superuser programmatically by creating the following environment variables in the .env.prod file:
* DJANGO_SUPERUSER_PASSWORD=enter_a_password
* DJANGO_SUPERUSER_EMAIL=example@email.com
* DJANGO_SUPERUSER_USERNAME=user_name
Then run the command:
```
sudo docker compose -f docker-compose.prod.yml exec web python manage.py createsuperuser --noinput
```
Or, you can do:
```
sudo docker compose -f docker-compose.prod.yml exec web python manage.py createsuperuser --username=joe --email=joe@example.com
```

## Acquire certificates from Let's Encrypt with certbot
Before acquiring the certificates:
- Comment out everything having to do with https in the Nginx config file.
- Comment out the Redirect to HTTPS location in the http section.

Do a dry run to test:
```
sudo docker compose -f docker-compose.prod.yml run --rm  certbot certonly --webroot --webroot-path /var/www/certbot/ --dry-run -d <domain.name>
```
And if it works, without the `--dry-run`:
```
sudo docker compose -f docker-compose.prod.yml run --rm  certbot certonly --webroot --webroot-path /var/www/certbot/ -d <domain.name>
```

After the successful acquiring of certificates:
- Uncomment everything having to do with https in the Nginx config file.
- Comment out the location which proxies to Django in the http section.
- Comment out the location with static files in the http section.
- Run the entire app with docker compose.

## Updating and restarting the server
### Pulling new production version from github
Upon pulling a new verstion of the project to the production server, you may have to reset all uncommited changes.
The below commands will remove all uncommitted changes, even if staged, and then pull:
```
git reset --hard HEAD
git pull
```
### Preparing the `nginx.conf` file
Make sure that the file is in a state suitable to whether you are recreating all the volumes, in which case the encryption certificates will be lost, and you have to acquire new with certbot. Or, if you simply are doing an update and leaving the volumes intact. See the comments in side the `nginx.conf` file for advice on which parts are necessary to activate / deactivate.

## Checklist for environment files before deployment
### Web app `.env` and `.env.prod` files
* `MAZU_ACTIVE` set to `True` or `False`, depending on whether AIs are active or not.

### `mazutalk.py` file
Adjust the variables for:
* `URL` set to local dev server or online production server.
* `LOOP`, `True` or `False`. Decides whether or not to run loop for connecting to web server.
* `N`, loop interval in seconds.
* `LOAD_MODEL`, `True` or `False`. Decides whether or not to load the model weights.
* `TRAIN`, `True` or `False`. Decides whether or not to train the model.
* `EPOCHS`, depending on how much training is desired.

### `mazusea.py` file
* `URL` set to local dev server or online production server.
* `N`, loop interval in seconds.

### `nginx.conf` file
Adjust which locations are used, depending on whether or not certificates are acquired.

### Order of actions for an online web server update
1. Stop the containers running.
2. Pull the latest update from github.
3. Start the app with docker compose up, specifying the production `docker-compose.prod.yml` file.
4. Do migrations with `makemigrations` and `migrate`.
5. Collect static files with `collectstatic`.
6. Recreate the superuser.

## Notes for creation of new apps and projects
### Beware of ownership when creating Django projects and apps
If you are running Docker on Linux, the files django-admin created are owned by root. This happens because the container runs as the root user. Change the ownership of the new files.
```
sudo chown -R $USER:$USER <files folders>
```

### Production security
To generate a secret key for Django:
```
python3 -c "import secrets; print(secrets.token_urlsafe())"
```

# Helpful articles
* https://testdriven.io/blog/dockerizing-django-with-postgres-gunicorn-and-nginx/
* https://github.com/docker/awesome-compose/tree/master/official-documentation-samples/django/
