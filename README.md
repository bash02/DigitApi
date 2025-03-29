git clone git@github.com:bash02/DigitApi.git
cd DigitApi

pipenv install  
pipenv shell

python manage.py migrate

python manage.py runserver
