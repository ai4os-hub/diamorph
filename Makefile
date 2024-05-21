ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# The env file is shared between this Makefile and docker compose
ENV_FILE=$(ROOT_DIR)/.env

# The envrc file is synced with the env file and is usable by direnv
ENVRC_FILE=$(ROOT_DIR)/.envrc

# Read the env file, and source it inside this Makefile
ifeq (,$(wildcard $(ENV_FILE)))
$(shell echo ROOT_DIR=${ROOT_DIR} > $(ENV_FILE))
$(shell cat .env.sample >> $(ENV_FILE))
$(info    )
$(info      an .env file has been generated for you; check its content)
$(info    )
endif

ifeq (,$(wildcard $(ENVRC_FILE)))
#$(shell echo 'test -f $(ENV_FILE) && source <(sed -e \'s/^[A-Z].*/export &/\' $(ENV_FILE))' > $(ENVRC_FILE))
$(shell echo 'test -f $(ENV_FILE) && source <(sed -e "s/^[A-Z].*/export &/" $(ENV_FILE))' > $(ENVRC_FILE))
$(info      an .envrc file has been generated for you; leave it as it is)
endif

# Import the content of the env file within the execution context of this Makefile
include $(ENV_FILE)
export $(shell sed -e 's/=.*//' -e 's/^\#.*//' $(ENV_FILE))

ifndef DOCKER_COMPOSE_USER
export DOCKER_COMPOSE_USER=$(shell id -un)
export DOCKER_COMPOSE_GROUP=$(shell id -gn)
export DOCKER_COMPOSE_UID=$(shell id -u)
export DOCKER_COMPOSE_GID=$(shell id -g)
endif

.DEFAULT_GOAL := help

.PHONY: help

OS=$(shell uname -s)

ifeq ($(OS),Linux)
INTERACTIVE=
OPEN=xdg-open
else ifeq ($(OS),Darwin)
INTERACTIVE=
OPEN=open
else ifeq ($(OS),CYGWIN_NT-10.0)
INTERACTIVE=winpty
OPEN=cmd /c start
else
INTERACTIVE=
OPEN=echo
endif

help: ## Display available commands in Makefile
	@grep -hE '^[a-zA-Z_0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## Build the images required by the application
	@docker compose build 

# This target force a direnv allow if the .env file has changed
.envrc: .env
	@touch .envrc

up: ## Creates and starts the application
	@docker compose up -d

stop: ## Stops the application
	@docker compose $@

start: ## Starts the application (must has been created before using the up target)
	@docker compose $@

restart : ## Restart the application
	@docker compose $@

ps: ## Show the running docker processes
	@docker compose $@

logs: ## Show the logs of the running docker processes
	@docker compose $@

rm: ## Remove containers
	@docker compose rm -sfv
