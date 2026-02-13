up:
	docker-compose up

retrain:
	docker compose run --rm -e FORCE_RETRAIN=true trainer

down:
	docker-compose down

logs:
	docker compose logs -f

build:
	docker compose build

rebuild:
	docker compose build --no-cache    