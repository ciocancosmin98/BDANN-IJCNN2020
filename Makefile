.PHONY: podman-image podman-run podman-clean podman-purge
	docker-image docker-run docker-clean docker-purge

TAG=ubuntu:bdann
CONTAINERNAME=bdann1

podman-image:
	podman build --tag $(TAG) ./Dockerfile
podman-run:
	podman run -it \
		--rm \
		--name=$(CONTAINERNAME) \
		--shm-size=4g \
		-v .:/storage \
		--security-opt=label=disable \
		$(TAG)
podman-clean:
	podman rmi $(TAG)
podman-purge:
	podman system prune --all --force && podman rmi --all

docker-image:
	docker build --tag $(TAG) .
docker-run:
	docker run -it \
		--rm \
		--name=$(CONTAINERNAME) \
		--shm-size=4g \
		-v "$(shell pwd)":/storage \
		--gpus all \
		$(TAG)
docker-clean:
	docker rmi $(TAG)
docker-purge:
	docker system prune --all --force
