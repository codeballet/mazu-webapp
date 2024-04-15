# Mazu Webapp

## Tips
### Run commands inside the container
```
docker compose exec web python manage.py makemigrations 
docker compose exec web python manage.py migrate
```

### Beware of ownership!
If you are running Docker on Linux, the files django-admin created are owned by root. This happens because the container runs as the root user. Change the ownership of the new files.
```
sudo chown -R $USER:$USER <files folders>
```

### Create `superuser` programmatically
Create superuser programmatically by creating the following environment variables in the .env file:
* DJANGO_SUPERUSER_PASSWORD=enter_a_password
* DJANGO_SUPERUSER_EMAIL=example@email.com
* DJANGO_SUPERUSER_USERNAME=user_name
Then run the command:
```
docker compose exec web python manage.py createsuperuser --noinput
```

# Guidance from
* https://testdriven.io/blog/dockerizing-django-with-postgres-gunicorn-and-nginx/
* https://github.com/docker/awesome-compose/tree/master/official-documentation-samples/django/
